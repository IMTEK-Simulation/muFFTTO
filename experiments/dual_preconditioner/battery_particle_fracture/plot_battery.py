"""Figures of the core-shell battery particle study (reads exp_data/battery_*_steps.csv, _iterations.csv,
_snapshots.npz written by example_2D_battery_particle_fracture_AT2.py).

    python plot_battery.py                                         # first series (12 um cell, l = 0.75 um)
    SUFFIX=_tu L_CELL_UM=6 GRIDS="64 128" python plot_battery.py   # Tu et al. length scales (run_battery_tu.sh)
"""
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data')
figs = os.path.join(here, 'figures')
os.makedirs(figs, exist_ok=True)
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 12})
plt.rcParams['axes.formatter.useoffset'] = False

PRECS = ['Green', 'Green_Jacobi', 'dual']
COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
SUFFIX = os.environ.get('SUFFIX', '')
L_CELL = float(os.environ.get('L_CELL_UM', 12.0))
GRIDS = [int(n) for n in os.environ.get('GRIDS', '32 64 128').split()]
GRID_STYLE = {32: (':', 's'), 64: ('--', 'o'), 128: ('-', '^')}
T_SCALE = (L_CELL * 1e-6) ** 2 / 3.26e-14          # s per time unit
saved = []


def load(N, P, ms='0.01'):
    base = os.path.join(data, f'battery_N{N}_{P}_ms{ms}{SUFFIX}')
    if not os.path.exists(base + '_steps.csv'):
        return None
    steps = np.genfromtxt(base + '_steps.csv', delimiter=',', names=True, dtype=None, encoding=None)
    its = np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None,
                        missing_values='', filling_values=-1)
    if steps.size < 2:
        return None
    snaps = np.load(base + '_snapshots.npz') if os.path.exists(base + '_snapshots.npz') else None
    return dict(steps=np.atleast_1d(steps), its=np.atleast_1d(its), snaps=snaps)


def save(fig, name):
    path = os.path.join(figs, name.replace('.png', f'{SUFFIX}.png'))
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved.append(path)
    print('saved', path)


def windows(steps):
    """(before, during) masks over the steps: before = the seed only (a_c within 0.002 of its first value),
    during = from the onset until a_c reaches 95 % of its final growth"""
    a = steps['a_c']
    growth = a - a[0]
    onset = growth > 0.002
    if not onset.any():
        return np.ones(len(a), bool), np.zeros(len(a), bool)
    i0 = np.argmax(onset)
    i1 = np.argmax(growth >= 0.95 * growth[-1])
    idx = np.arange(len(a))
    return idx < i0, (idx >= i0) & (idx <= max(i1, i0))


runs = {(N, P): load(N, P) for N in GRIDS for P in PRECS}

# ---------------------------------------------------------------- 1. snapshots of c and d (finest dual run available)
for N in GRIDS[::-1]:
    r = runs[(N, 'dual')]
    if r is not None and r['snaps'] is not None:
        z = r['snaps']
        t = z['time']
        st = r['steps']                     # snapshot times through the crack growth: a_c at 0, 1/4, ... of its growth
        g = st['a_c'] - st['a_c'][0]
        targets = [st['time'][0]] + [st['time'][np.argmax(g >= f * g[-1])] for f in (0.02, 0.25, 0.5, 0.95)] \
            + [st['time'][-1]]
        idx = np.unique([int(np.argmin(np.abs(t - tt))) for tt in targets])
        ext = (0, L_CELL, 0, L_CELL)
        fig, axes = plt.subplots(2, len(idx), figsize=(3.6 * len(idx) + 1.2, 7.2), constrained_layout=True)
        axes = np.atleast_2d(axes).reshape(2, -1)
        for k, i in enumerate(idx):
            c = np.where(z['particle'] > 0, z['concentration'][i], np.nan)
            im_c = axes[0, k].imshow(c.T, origin='lower', interpolation='nearest', extent=ext, vmin=0.1, vmax=1.0,
                                     cmap='viridis')
            im_d = axes[1, k].imshow(z['damage'][i].T, origin='lower', interpolation='nearest', extent=ext,
                                     vmin=0, vmax=1, cmap='inferno')
            axes[0, k].set_title(f't = {t[i] * T_SCALE:.0f} s')
            for a in axes[:, k]:
                a.set_xlabel(r'x [$\mu$m]')
                a.set_ylabel(r'y [$\mu$m]')
        fig.colorbar(im_c, ax=axes[0, :], label='filling c / c_max', shrink=0.9)
        fig.colorbar(im_d, ax=axes[1, :], label='damage d', shrink=0.9)
        fig.suptitle(f'core-shell particle, lithiation 1C, {N}x{N} pixels (dual run)')
        save(fig, f'battery_snapshots_N{N}.png')
        break

# ---------------------------------------------------------------- 2. crack growth and SOL vs time
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)
for N in GRIDS:
    for P in PRECS:
        r = runs[(N, P)]
        if r is None:
            continue
        s = r['steps']
        ls, mk = GRID_STYLE[N]
        kw = dict(color=COLOR[P], ls=ls, lw=1.6 if P == 'dual' else 1.0, label=f'{N}$^2$ {LABEL[P]}')
        axes[0].plot(s['time'] * T_SCALE, s['a_c'], **kw)
        axes[1].plot(s['time'] * T_SCALE, s['gamma'] * L_CELL, **kw)
        axes[2].plot(s['time'] * T_SCALE, s['sol'], **kw)
axes[0].set_ylabel(r'broken area fraction $\bar a_c$ (d > 0.95)')
axes[1].set_ylabel(r'AT2 crack length $\Gamma$ [$\mu$m]')
axes[2].set_ylabel('state of lithiation SOL')
for a in axes:
    a.set_xlabel('time [s]')
    a.set_xlim(left=0)
    a.set_ylim(bottom=0)
    a.grid(alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center')
save(fig, 'battery_crack_growth.png')

# ---------------------------------------------------------------- 3. CG per elastic solve vs time step, per grid
avail = [N for N in GRIDS if any(runs[(N, P)] is not None for P in PRECS)]
fig, axes = plt.subplots(1, len(avail) + 1, figsize=(5.2 * (len(avail) + 1), 4.6), constrained_layout=True)
for ax, N in zip(axes, avail):
    for P in PRECS:
        r = runs[(N, P)]
        if r is None:
            continue
        s = r['steps']
        ax.plot(s['step'], s['cg_u_mean'], color=COLOR[P], lw=1.3, label=LABEL[P])
    ax.set_title(f'elasticity, {N}x{N} (h = {L_CELL / N:.3g} $\\mu$m)')
    ax.set_xlabel('time step')
    ax.set_ylabel('mean CG iterations per elastic solve')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
ax = axes[-1]
for N in avail:
    r = runs[(N, 'dual')]
    if r is None:
        continue
    its = r['its']
    ls, mk = GRID_STYLE[N]
    for P in PRECS:
        col = its[f'cg_c_{P}'].astype(float)
        steps = its['step']
        per_step = np.array([col[steps == k].mean() for k in np.unique(steps)])
        ax.plot(np.unique(steps), per_step, color=COLOR[P], ls=ls, lw=1.2, label=f'{N}$^2$ {LABEL[P]}')
ax.set_title('diffusion (same trajectory, shadow solves)')
ax.set_xlabel('time step')
ax.set_ylabel('mean CG iterations per diffusion solve')
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(ncol=1, fontsize=10)
save(fig, 'battery_cg_per_step.png')

# ---------------------------------------------------------------- 4. mean CG per solve vs grid, before / during cracking
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
table = []
for P in PRECS:
    for w, (ax, title) in enumerate(zip(axes, ['before crack growth (seed only)', 'during crack growth'])):
        xs, ys = [], []
        for N in GRIDS:
            r = runs[(N, P)]
            if r is None:
                continue
            ref = runs[(N, 'dual')] if runs[(N, 'dual')] is not None else r   # windows of the (complete) dual run
            masks = windows(ref['steps'])
            steps_in = ref['steps']['step'][masks[w]]
            sel = np.isin(r['its']['step'], steps_in)
            if sel.any():
                xs.append(N)
                ys.append(r['its']['cg_u'][sel].mean())
                table.append((P, N, title, ys[-1], int(sel.sum())))
        ax.plot(xs, ys, color=COLOR[P], marker='o', lw=1.5, label=LABEL[P])
        ax.set_title(title)
for ax in axes:
    ax.set_xlabel('pixels per side N')
    ax.set_ylabel('mean CG iterations per elastic solve')
    ax.set_xticks(GRIDS)
    ax.set_xlim(0, 1.06 * GRIDS[-1])
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
save(fig, 'battery_cg_vs_grid.png')
for row in table:
    print(f'{row[0]:13s} N={row[1]:4d} {row[2]:32s} mean CG/solve = {row[3]:7.1f}  ({row[4]} solves)')
for N in GRIDS:                      # diffusion: shadow solves of the dual run, same windows
    r = runs[(N, 'dual')]
    if r is None:
        continue
    for w, name in enumerate(['before', 'during']):
        sel = np.isin(r['its']['step'], r['steps']['step'][windows(r['steps'])[w]])
        print(f'diffusion N={N:4d} {name:7s} mean CG/solve ' + '  '.join(
            f'{LABEL[P]} {r["its"][f"cg_c_{P}"][sel].mean():.1f}' for P in PRECS)
            + f'   damage {r["its"]["cg_d"][sel].mean():.1f}')

# ---------------------------------------------------------------- 5. matrix contrast (64^2)
ms_list = ['0.1', '0.01', '0.001']
cont = {(ms, P): load(64, P, ms) for ms in ms_list for P in PRECS}
if sum(v is not None for v in cont.values()) > 3:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for P in PRECS:
        for w, ax in enumerate(axes):
            xs, ys = [], []
            for i, ms in enumerate(ms_list):
                r = cont[(ms, P)]
                if r is None:
                    continue
                ref = cont[(ms, 'dual')] if cont[(ms, 'dual')] is not None else r
                masks = windows(ref['steps'])
                sel = np.isin(r['its']['step'], ref['steps']['step'][masks[w]])
                if sel.any():
                    xs.append(i)
                    ys.append(r['its']['cg_u'][sel].mean())
                    print(f'contrast {ms:6s} {P:13s} window {w}: {ys[-1]:.1f}')
            ax.plot(xs, ys, color=COLOR[P], marker='o', lw=1.5, label=LABEL[P])
    for ax, title in zip(axes, ['before crack growth', 'during crack growth']):
        ax.set_xticks(range(len(ms_list)))
        ax.set_xticklabels(ms_list)
        ax.set_xlabel('matrix stiffness / core stiffness')
        ax.set_ylabel('mean CG iterations per elastic solve')
        ax.set_title(f'64x64, {title}')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, 'battery_cg_vs_matrix_contrast.png')

for p in saved:
    os.system(f'xdg-open {p} >/dev/null 2>&1 &')
