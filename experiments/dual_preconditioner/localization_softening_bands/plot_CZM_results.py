"""Figures of the PF-CZM concrete study (exp_data/CZM_*.csv/.npz -> figures/).

python plot_CZM_results.py [LOAD_DIR=xx] [K_RESIDUAL=0.001] [RUN_SUFFIX]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
load_dir = sys.argv[1] if len(sys.argv) > 1 else 'xx'
k_res = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
tag = f'_asr_{load_dir}_cornelissen_b5mm_k{k_res:g}' + (sys.argv[3] if len(sys.argv) > 3 else '')
E_UNIT = 30e3
grids = [32, 64, 128]
precs = ['Green', 'Green_Jacobi', 'dual']
color = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
label = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
marker = {32: 'o', 64: 's', 128: '^'}
lstyle = {32: ':', 64: '--', 128: '-'}
comp = r'\bar\varepsilon_{xx}' if load_dir == 'xx' else r'\bar\gamma_{xy}'
scomp = r'\bar\sigma_{xx}' if load_dir == 'xx' else r'\bar\sigma_{xy}'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13})
fig_dir = os.path.join(here, 'figures')
os.makedirs(fig_dir, exist_ok=True)


def load(N, P):
    base = os.path.join(here, 'exp_data', f'CZM_N{N}_{P}{tag}')
    if not os.path.exists(base + '_steps.csv'):
        return None
    st = np.genfromtxt(base + '_steps.csv', delimiter=',', names=True)
    it = np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True)
    if st.size < 2:
        return None
    return st, it, base


def save(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('saved', path)


def scale_ticks(ax, scale):
    """plain tick labels for strains/stresses: values x 10^3 shown"""
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v * scale:g}'))


data = {(N, P): load(N, P) for N in grids for P in precs}

# 1. stress-strain and minimum band stiffness
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for N in grids:
    for P in precs:
        d = data[(N, P)]
        if d is None:
            continue
        st = d[0]
        lab = f'{label[P]} {N}²'
        axes[0].plot(np.r_[0, st['load']], np.r_[0, st['stress']] * E_UNIT, lstyle[N], color=color[P], label=lab, lw=1.6)
        axes[1].plot(st['load'], st['crack_fraction_09'], lstyle[N], color=color[P], lw=1.6)
axes[0].set_xlabel(rf'macroscopic strain ${comp}$ ($\times 10^{{-3}}$)')
axes[0].set_ylabel(rf'macroscopic stress ${scomp}$ (MPa)')
axes[0].set_xlim(0)
axes[0].set_ylim(0)
axes[0].set_title('stress-strain (all preconditioners overlap)')
axes[1].set_xlabel(rf'macroscopic strain ${comp}$ ($\times 10^{{-3}}$)')
axes[1].set_ylabel(r'cracked area fraction ($d > 0.9$)')
axes[1].set_xlim(0)
axes[1].set_title('crack growth (d > 0.9)')
axes[1].set_ylim(0)
for ax in axes:
    scale_ticks(ax, 1e3)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=9, ncol=1, loc='upper right')
fig.tight_layout()
save(fig, f'CZM_stress_strain{tag}.png')

# 2. mean CG per elastic solve in each load step vs load
avail = [N for N in grids if any(data[(N, P)] is not None for P in precs)]
fig, axes = plt.subplots(1, len(avail), figsize=(5.2 * len(avail), 4.6), squeeze=False)
for ax, N in zip(axes[0], avail):
    for P in precs:
        d = data[(N, P)]
        if d is None:
            continue
        st = d[0]
        ax.plot(st['load'], st['mean_cg_u'], '-', color=color[P], label=label[P], lw=1.6)
        ipeak = int(np.argmax(st['stress']))
    ax.axvline(st['load'][ipeak], color='k', lw=0.8, ls=':')
    ax.set_xlabel(rf'${comp}$ ($\times 10^{{-3}}$)')
    ax.set_ylabel('mean CG iterations per elastic solve')
    ax.set_title(f'{N}², b/h = {0.0625 * N:g}')
    ax.set_xlim(0)
    ax.set_ylim(0)
    scale_ticks(ax, 1e3)
    ax.grid(alpha=0.3)
axes[0][0].legend()
fig.tight_layout()
save(fig, f'CZM_cg_per_solve{tag}.png')


# 3. mean CG per solve vs grid: before the peak / softening (after the peak)
def phase_means(d):
    st, it, _ = d
    peak_load = st['load'][int(np.argmax(st['stress']))]
    pre = it['cg_u'][it['load'] <= peak_load + 1e-12]
    post = it['cg_u'][it['load'] > peak_load + 1e-12]
    return pre.mean(), (post.mean() if post.size else np.nan), it['cg_u'].sum()


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
rows = []
for P in precs:
    ns, pre, post, tot = [], [], [], []
    for N in grids:
        d = data[(N, P)]
        if d is None:
            continue
        a, b, c = phase_means(d)
        ns.append(N); pre.append(a); post.append(b); tot.append(c)
        rows.append((P, N, a, b, c, len(d[1]), int(d[0]['converged'].size - d[0]['converged'].sum()), d[1]['time_u'].sum()))
    axes[0].plot(ns, pre, 'o--', color=color[P], label=f'{label[P]}, before peak', mfc='white', ms=8)
    axes[0].plot(ns, post, 's-', color=color[P], label=f'{label[P]}, softening', ms=8)
    axes[1].plot(ns, np.array(tot) / 1e3, 's-', color=color[P], label=label[P], ms=8)
for ax in axes:
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}²'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('grid (fixed b = 5 mm)')
    ax.set_ylim(0)
    ax.grid(alpha=0.3)
axes[0].set_ylabel('mean CG iterations per elastic solve')
axes[0].set_title('per solve: before peak vs softening')
axes[1].set_ylabel(r'total elastic CG iterations ($\times 10^3$)')
axes[1].set_title('whole load path (dual: 2 Green appl./it)')
axes[0].legend(fontsize=10)
axes[1].legend(fontsize=10)
fig.tight_layout()
save(fig, f'CZM_cg_vs_grid{tag}.png')
print('P, N, mean CG pre-peak, mean CG softening, total CG, elastic solves, unconverged steps, elastic time [s]')
for r in rows:
    print(f'{r[0]:13s} {r[1]:4d} {r[2]:7.1f} {r[3]:7.1f} {int(r[4]):8d} {r[5]:6d} {r[6]:3d} {r[7]:8.1f}')

# 4. damage snapshots (dual run) at the finest available grid
for N in avail:
    P = 'dual' if data[(N, 'dual')] is not None else next(p for p in precs if data[(N, p)] is not None)
    f = np.load(data[(N, P)][2] + '_fields.npz')
    D, loads, stress, matrix = f['d'], f['load'], f['stress'], f['phase']
    ipeak = int(np.argmax(stress))
    picks = sorted(set([max(ipeak - 2, 0), ipeak, ipeak + (len(loads) - ipeak) // 8, ipeak + (len(loads) - ipeak) // 4,
                        ipeak + (len(loads) - ipeak) // 2, len(loads) - 1]))
    picks = [p for p in picks if p < len(loads)]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.9 * len(picks), 4.0), layout='constrained')
    cmap = plt.get_cmap('inferno').copy()
    cmap.set_bad('0.65')
    for ax, p in zip(axes, picks):
        im = ax.imshow(D[p].T, origin='lower', extent=(0, 80, 0, 80), vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
        ax.contour(np.linspace(0, 80, N, endpoint=False) + 40 / N, np.linspace(0, 80, N, endpoint=False) + 40 / N, matrix.T, [0.5], colors='cyan', linewidths=0.7)
        ax.set_title(rf'${comp}$ = {loads[p] * 1e3:.2f}e-3' + ('\n(peak)' if p == ipeak else
                     f'\nσ = {stress[p] * E_UNIT:.2f} MPa'), fontsize=11)
        ax.set_xticks([0, 40, 80])
        ax.set_yticks([0, 40, 80])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=axes, shrink=0.8, label='phase field d (cyan: aggregate boundary)')
    fig.suptitle(f'PF-CZM (Cornelissen), {N}², {label[P]}, k = {k_res:g}, b = 5 mm, cell 80 mm', fontsize=13)
    save(fig, f'CZM_damage_snapshots_N{N}{tag}.png')

# 5. mean CG per elastic solve binned by the current minimum band stiffness (same softening level on every grid)
bins = [(1.01, 0.1), (0.1, 0.01), (0.01, 2e-3), (2e-3, 0)]
print('mean CG per solve by min band stiffness s: ' + ', '.join(f'[{b:g},{a:g})' for a, b in bins))
for P in precs:
    for N in grids:
        d = data[(N, P)]
        if d is None:
            continue
        s, cg = d[1]['min_s_matrix'], d[1]['cg_u']
        vals = [cg[(s < a) & (s >= b)] for a, b in bins]
        print(f'{P:13s} {N:4d} ' + ' '.join(f'{v.mean():7.1f}' if v.size else '      -' for v in vals))
