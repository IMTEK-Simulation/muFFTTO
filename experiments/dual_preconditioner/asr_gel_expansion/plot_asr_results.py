"""
Figures of the ASR study (exp_data/ASR_N*_<P>_s1_l<l>_steps.npz / _iterations.csv) -> figures/
    python plot_asr_results.py            env: LENGTH_SCALE (0.0625), GRIDS ('32,64,128'), SUFFIX ('')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
l = float(os.environ.get('LENGTH_SCALE', 0.0625))
grids = [int(v) for v in os.environ.get('GRIDS', '32,64,128').split(',')]
suffix = os.environ.get('SUFFIX', '')
PREC = {'Green': ('Green', 'tab:green', 'o', '-'), 'Green_Jacobi': ('Green-Jacobi', 'tab:blue', 's', '--'),
        'dual': ('dual', 'tab:red', '^', '-.')}
D_CRACK = 0.9           # a load step counts as 'cracked' once max d >= D_CRACK
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 12})
fig_dir = os.path.join(here, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def name(N, P):
    return os.path.join(here, 'exp_data', f'ASR_N{N}_{P}_s1_l{l:g}{suffix}')


def load(N, P):
    if not os.path.exists(name(N, P) + '_iterations.csv'):
        return None, None
    it = np.genfromtxt(name(N, P) + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None)
    it = np.atleast_1d(it)
    st = np.load(name(N, P) + '_steps.npz') if os.path.exists(name(N, P) + '_steps.npz') else None
    return it, st


def cg4(it):
    """mean CG iterations of the 4 elastic solves (3 unit strains + eigenstrain) of each staggered iteration"""
    return (it['cg_gel'] + it['cg_unit_xx'] + it['cg_unit_yy'] + it['cg_unit_xy']) / 4.0


def save(fig, fname):
    path = os.path.join(fig_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


data = {(N, P): load(N, P) for N in grids for P in PREC}
cmap3 = ListedColormap(['#d9cfa8', '#7f7f7f', '#1f3fbf'])     # paste, aggregate, gel

# ---- geometry on the exact pixels of every grid
fig, axes = plt.subplots(1, len(grids), figsize=(4.6 * len(grids), 4.6), layout='constrained')
for ax, N in zip(np.atleast_1d(axes), grids):
    st = next((data[(N, P)][1] for P in PREC if data[(N, P)][1] is not None), None)
    if st is None:
        continue
    ax.imshow(st['label'].T, origin='lower', cmap=cmap3, vmin=-0.5, vmax=2.5, interpolation='nearest',
              extent=(0, 1, 0, 1))
    ax.set_title(f'{N}$^2$ (l/h = {l * N:g})')
    ax.set_xlabel('x / 0.1 m')
    ax.set_ylabel('y / 0.1 m')
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap3.colors]
fig.legend(handles, ['cement paste', 'aggregate', 'ASR gel pocket'], loc='lower center', ncol=3,
           bbox_to_anchor=(0.5, -0.1))
save(fig, f'asr_geometry{suffix}_l{l:g}.png')

# ---- damage snapshots vs gel expansion (dual), one row per grid
levels = None
rows = [N for N in grids if data[(N, 'dual')][1] is not None]
if rows:
    dvv_all = data[(rows[-1], 'dual')][1]['dVV']
    levels = np.linspace(0, float(os.environ.get('DVV_MAX', 0.024)), 5)
    fig, axes = plt.subplots(len(rows), len(levels), figsize=(4.0 * len(levels), 3.8 * len(rows)), squeeze=False,
                             layout='constrained')
    for r, N in enumerate(rows):
        st = data[(N, 'dual')][1]
        dvv, dmg = st['dVV'], st['damage']               # damage[k] belongs to dVV[k + 1]
        gel = st['label'] == 2
        for c, lev in enumerate(levels):
            ax = axes[r, c]
            if lev == 0:
                img = np.zeros(gel.shape)
            else:
                k = int(np.argmin(np.abs(dvv[1:] - lev)))
                if abs(dvv[1 + k] - lev) > 1e-9:
                    ax.set_axis_off()
                    continue
                img = np.clip(dmg[k], 0, 1)
            im = ax.imshow(img.T, origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest',
                           extent=(0, 1, 0, 1))
            ax.imshow(np.where(gel, 1.0, np.nan).T, origin='lower', cmap=ListedColormap(['#40c0ff']),
                      interpolation='nearest', extent=(0, 1, 0, 1))
            ax.contour(np.linspace(0.5 / N, 1 - 0.5 / N, N), np.linspace(0.5 / N, 1 - 0.5 / N, N),
                       (st['label'] == 1).T.astype(float), levels=[0.5], colors='w', linewidths=0.6)
            ax.set_title(f'{N}$^2$, $\\Delta V/V$ = {lev:.3f}')
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel('x / 0.1 m')
            ax.set_ylabel('y / 0.1 m')
    fig.colorbar(im, ax=axes, shrink=0.6, label='damage d')
    fig.suptitle(f'dual preconditioner run, l = {l:g} (x 0.1 m); cyan: gel pockets, white: aggregate boundaries')
    save(fig, f'asr_damage_evolution_dual{suffix}_l{l:g}.png')

# ---- stiffness reduction vs expansion
fig, ax = plt.subplots(figsize=(6.5, 4.5))
# the damage path is the same for every preconditioner (CG to 1e-6): one curve per grid, the largest deviation printed
for N, ls, shade in zip(grids, ['-', '--', ':'], ['0.0', '0.35', '0.6']):
    runs = [data[(N, P)][1] for P in PREC if data[(N, P)][1] is not None]
    if not runs:
        continue
    n = min(len(r['dVV']) for r in runs)
    dev = max(np.abs(r['stiffness_ratio'][:n] - runs[0]['stiffness_ratio'][:n]).max() for r in runs)
    print(f'{N}^2: max |stiffness ratio difference| between preconditioners = {dev:.2e}')
    ax.plot(runs[0]['dVV'], runs[0]['stiffness_ratio'], color=shade, ls=ls, lw=2, label=f'{N}$^2$, l/h = {l * N:g}')
if suffix == '_real':      # Falsafi et al. Fig. 8a (crack band, digitised by eye, +-0.02): h = 1.56 / 0.78 / 0.39 mm
    xs = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 6.4]) / 100
    for ys, ls, lab in (([1, 0.94, 0.86, 0.76, 0.67, 0.54, 0.45, 0.37, 0.32, 0.30], '-', '64$^2$'),
                        ([1, 0.93, 0.84, 0.74, 0.645, 0.52, 0.42, 0.35, 0.30, 0.285], '--', '128$^2$'),
                        ([1, 0.94, 0.85, 0.75, 0.66, 0.54, 0.45, 0.38, 0.34, 0.32], ':', '256$^2$')):
        ax.plot(xs, ys, color='tab:orange', ls=ls, marker='o', ms=3, label=f'Falsafi et al. Fig. 8a, {lab}')
ax.set_xlabel(r'gel area expansion $\Delta V / V$')
ax.set_ylabel(r'$\|C_\mathrm{eff}\| \,/\, \|C_\mathrm{eff,0}\|$')
ax.set_xlim(0, None)
ax.set_ylim(0, 1.05)
ax.legend()
ax.set_title('effective stiffness (all preconditioners)')
ax.grid(alpha=0.3)
save(fig, f'asr_stiffness_reduction{suffix}_l{l:g}.png')

# ---- CG per elastic solve vs load step (mean over the 4 solves of all staggered iterations of the step)
fig, axes = plt.subplots(1, len(grids), figsize=(5.2 * len(grids), 4.3), squeeze=False)
for ax, N in zip(axes[0], grids):
    for P, (lab, col, mk, ls) in PREC.items():
        it = data[(N, P)][0]
        if it is None:
            continue
        cg_u = cg4(it)
        steps = np.unique(it['step'])
        ax.plot([it['dVV'][it['step'] == s][0] for s in steps], [cg_u[it['step'] == s].mean() for s in steps],
                color=col, marker=mk, ms=3, ls=ls, label=lab)
    ax.set_title(f'{N}$^2$, l/h = {l * N:g}')
    ax.set_xlabel(r'gel area expansion $\Delta V / V$')
    ax.set_ylabel('CG iterations per elastic solve')
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
axes[0, 0].legend()
save(fig, f'asr_cg_per_solve_vs_load{suffix}_l{l:g}.png')

# ---- mean CG per elastic solve vs grid: before cracking (max d < D_CRACK) and during crack growth
table = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for P, (lab, col, mk, ls) in PREC.items():
    pre, post, Ns = [], [], []
    for N in grids:
        it = data[(N, P)][0]
        if it is None:
            continue
        # only the load steps completed by all three runs on this grid (a fair comparison of unfinished runs)
        last = min(int(data[(N, Q)][0]['step'].max()) for Q in PREC if data[(N, Q)][0] is not None)
        it = it[it['step'] <= last]
        cg = cg4(it)
        cracked = it['max_d'] >= D_CRACK
        pre.append(cg[~cracked].mean() if (~cracked).any() else np.nan)
        post.append(cg[cracked].mean() if cracked.any() else np.nan)
        Ns.append(N)
        table.append((P, N, last, pre[-1], post[-1], len(it), int(it['staggered_iteration'].max()),
                      it['time_u'].sum() + it['time_d'].sum()))
    for ax, vals in zip(axes, (pre, post)):
        ax.plot(Ns, vals, color=col, marker=mk, ls=ls, label=lab, ms=7)
for ax, title in zip(axes, (f'before cracking (max d < {D_CRACK})', f'crack growth (max d $\\geq$ {D_CRACK})')):
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids)
    ax.set_xticklabels([f'{N}$^2$' for N in grids])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlabel('grid')
    ax.set_ylabel('mean CG iterations per elastic solve')
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[0].legend()
save(fig, f'asr_cg_vs_grid{suffix}_l{l:g}.png')
print(f'{"precond":14s} {"N":>4s} {"steps":>5s} {"CG pre":>8s} {"CG crack":>9s} {"stagg its":>9s} {"max/step":>8s} {"time s":>8s}')
for P, N, last, a, b, n, m, t in table:
    print(f'{P:14s} {N:4d} {last + 1:5d} {a:8.1f} {b:9.1f} {n:9d} {m:8d} {t:8.0f}')
