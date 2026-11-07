"""
Figures of fourel_nonlocal_damage.py runs (exp_data/FOUREL_*), written to figures/FOUREL_*.png.

    python plot_fourel.py            # env NS (64,128,256), LOADING (free), K (0.0001), SUFFIX (''), SNAP_SOLVER (Green),
                                     #     MS_TAGS (',_tolms1e-06': MS variants to show), EXTRA ('' -> e.g. _batch)
Cost unit = one Green-preconditioned CG iteration = one MS fixed-point iteration (1 B, 1 B^T, 1 FFT pair);
Green-Jacobi = 1, dual = 2 (two Green applications + one K product per iteration).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data')
figs = os.path.join(here, 'figures')
os.makedirs(figs, exist_ok=True)
plt.rcParams.update({'font.size': 13, 'axes.formatter.useoffset': False})

NS = [int(n) for n in os.environ.get('NS', '64,128,256').split(',')]
loading = os.environ.get('LOADING', 'free')
K = os.environ.get('K', '0.0001')
suffix = os.environ.get('SUFFIX', '')
extra = os.environ.get('EXTRA', '')
snap_solver = os.environ.get('SNAP_SOLVER', 'Green')
ms_tags = os.environ.get('MS_TAGS', ',_tolms1e-06').split(',')
COST = {'MS': 1.0, 'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}
STYLE = {'Green': dict(color='tab:green', ls='-', marker='o', label='Newton + Green'),
         'Green_Jacobi': dict(color='tab:blue', ls='-', marker='s', label='Newton + Green-Jacobi'),
         'dual': dict(color='tab:red', ls='-', marker='^', label='Newton + dual'),
         'MS': dict(color='black', ls='-', marker='x', label='MS basic scheme (tol 1e-4)'),
         'MS_tolms1e-05': dict(color='dimgrey', ls='--', marker='+', label='MS basic scheme (tol 1e-5)'),
         'MS_tolms1e-06': dict(color='grey', ls=':', marker='d', label='MS basic scheme (tol 1e-6)')}


def run_key(solver, tag=''):
    return solver + tag


def load(N, solver, tag=''):
    base = os.path.join(data, f'FOUREL_N{N}_{solver}_{loading}_k{K}{extra}{tag}{suffix}')
    if not os.path.exists(base + '_steps.npz'):
        return None
    st = dict(np.load(base + '_steps.npz', allow_pickle=True))
    sv = np.atleast_1d(np.genfromtxt(base + '_solves.csv', delimiter=',', names=True, dtype=None, encoding=None))
    sv = sv[sv['step'] < len(st['eps_asr'])]                   # completed load steps only (running jobs)
    return dict(steps=st, solves=sv, base=base)


runs = {}
for N in NS:
    for s in ('Green', 'Green_Jacobi', 'dual'):
        r = load(N, s)
        if r is not None:
            runs[(N, s)] = r
    for t in ms_tags:
        r = load(N, 'MS', t)
        if r is not None:
            runs[(N, 'MS' + t)] = r
print('runs:', sorted(runs))


def cost(key, r, eps_end=None):
    """cost units up to (and including) the load step with eps_ASR = eps_end (default: all); unit solves from the
    steps file (step-start C_eff + recomputations inside the solves)"""
    c = COST[key[1].split('_tolms')[0]]
    sv, st = r['solves'], r['steps']
    e = st['eps_asr'][-1] if eps_end is None else eps_end
    return c * (sv['linear_its'][sv['eps_asr'] <= e + 1e-12].sum() + st['unit_its'][st['eps_asr'] <= e + 1e-12].sum())


def common_end(N):
    return min(runs[k]['steps']['eps_asr'][-1] for k in runs if k[0] == N)


# ---------------------------------------------------------------- table
lines = ['N  solver  steps  eps_end%  solves  spots  linear_its  unit_its  its/solve  cost_units  wall_s  '
         'energy_err_median']
for key in sorted(runs, key=lambda k: (k[0], k[1])):
    r = runs[key]
    sv, st = r['solves'], r['steps']
    e = sv['energy_error'][~np.isnan(sv['energy_error'])]
    lines.append(f'{key[0]} {key[1]} {len(st["eps_asr"]) - 1} {100 * st["eps_asr"][-1]:.3f} {len(sv)} '
                 f'{int(sv["spots_total"].max())} {int(sv["linear_its"].sum())} {int(st["unit_its"].sum())} '
                 f'{sv["linear_its"].mean():.1f} {cost(key, r):.0f} {st["time_step"].sum():.0f} '
                 f'{(np.median(e) if e.size else np.nan):.1e}')
table = '\n'.join(lines)
print(table)
with open(os.path.join(figs, f'FOUREL_cost_table_{loading}{extra}{suffix}.txt'), 'w') as f:
    f.write(table + '\n')


def series_per_step(r, field):
    sv = r['solves']
    steps = np.unique(sv['step']).astype(int)
    eps = np.array([sv['eps_asr'][sv['step'] == s][0] for s in steps])
    val = np.array([sv[field][sv['step'] == s].mean() for s in steps])
    return eps, val


def show(path):
    print('->', path)
    if os.environ.get('OPEN', '0') == '1':                     # intermediate figures: not opened by default
        os.system(f'xdg-open {path} >/dev/null 2>&1 &')


# ---------------------------------------------------------------- iterations per solve vs eps_ASR, cumulative cost
for N in NS:
    keys = [k for k in runs if k[0] == N]
    if not keys:
        continue
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for k in sorted(keys, key=lambda k: k[1]):
        r = runs[k]
        st = STYLE[k[1]]
        eps, its = series_per_step(r, 'linear_its')
        ax[0].plot(100 * eps, its, ls=st['ls'], color=st['color'], marker=st['marker'], ms=4, markevery=10,
                   label=st['label'])
        sv = r['solves']
        unit_first = np.zeros(len(sv))                        # step-start unit solves on the first solve of the step
        su = r['steps']['unit_its'] - np.array([sv['unit_its'][sv['step'] == j].sum()
                                                for j in range(len(r['steps']['unit_its']))])
        for j, v in enumerate(su):
            idx = np.flatnonzero(sv['step'] == j)
            if idx.size:
                unit_first[idx[0]] = v
        c = COST[k[1].split('_tolms')[0]] * np.cumsum(sv['linear_its'] + sv['unit_its'] + unit_first)
        ax[1].plot(100 * sv['eps_asr'], c / 1e3, ls=st['ls'], color=st['color'], marker=st['marker'], ms=4,
                   markevery=max(1, len(c) // 10), label=st['label'])
    ax[0].set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
    ax[0].set_ylabel('iterations per elastic solve\n(mean over the load step)')
    ax[0].set_title(f'{N}$^2$: cost of one elastic solve')
    ax[1].set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
    ax[1].set_ylabel('cumulative cost [10$^3$ Green-iteration units]')
    ax[1].set_title(f'{N}$^2$: cumulative cost (dual = 2 units / it)')
    for a in ax:
        a.set_xlim(left=0)
        a.set_ylim(bottom=0)
        a.grid(alpha=0.3)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    p = os.path.join(figs, f'FOUREL_cost_N{N}_{loading}{extra}{suffix}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    show(p)

# ---------------------------------------------------------------- totals bar chart
order = ['MS', 'MS_tolms1e-05', 'MS_tolms1e-06', 'Green', 'Green_Jacobi', 'dual']
present = [o for o in order if any(k[1] == o for k in runs)]
Ns = [N for N in NS if any(k[0] == N for k in runs)]
if present and Ns:
    fig, ax = plt.subplots(1, len(Ns), figsize=(5.2 * len(Ns), 5.2), squeeze=False)
    for i, N in enumerate(Ns):
        a = ax[0, i]
        for j, o in enumerate(present):
            if (N, o) in runs:
                r = runs[(N, o)]
                c = cost((N, o), r, common_end(N))
                a.bar(j, c / 1e3, color=STYLE[o]['color'], hatch='//' if 'tolms' in o else None,
                      edgecolor='white' if 'tolms' not in o else 'black')
                a.text(j, c / 1e3, f'{c / 1e3:.0f}', ha='center', va='bottom', fontsize=11)
        a.set_xticks(range(len(present)))
        a.set_xticklabels([o.replace('_tolms', '\ntol ').replace('Green_Jacobi', 'Green-\nJacobi') for o in present],
                          fontsize=11)
        a.set_ylabel('total cost [10$^3$ Green-iteration units]')
        eps_end = 100 * common_end(N)
        a.set_title(f'{N}$^2$, up to $\\varepsilon_{{ASR}}$ = {eps_end:.2f} %')
        a.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = os.path.join(figs, f'FOUREL_cost_totals_{loading}{extra}{suffix}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    show(p)

# ---------------------------------------------------------------- macro expansion, stiffness
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
for N in NS:
    for s in ('Green', 'MS'):
        if (N, s) not in runs:
            continue
        st = runs[(N, s)]['steps']
        E = st['E']
        ls = {64: ':', 128: '--', 256: '-'}[N]
        col = STYLE[s]['color']
        ax[0].plot(100 * st['eps_asr'], 100 * (E[:, 0] + E[:, 1]), ls=ls, color=col, label=f'{N}$^2$ {STYLE[s]["label"]}')
        if s == 'Green':
            C = st['C_eff']
            ax[1].plot(100 * st['eps_asr'], C[:, 0, 0] / 1e3, ls=ls, color=col, label=f'{N}$^2$ $C_{{xxxx}}$')
            ax[1].plot(100 * st['eps_asr'], C[:, 1, 1] / 1e3, ls=ls, color='tab:olive', label=f'{N}$^2$ $C_{{yyyy}}$')
ax[0].set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
ax[0].set_ylabel(r'macroscopic expansion tr $\bar\varepsilon$ [%]')
ax[0].set_title('macroscopic (areal) expansion')
ax[1].set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
ax[1].set_ylabel('effective tangent stiffness [GPa]')
ax[1].set_title('effective tangent stiffness (Newton runs)')
for a in ax:
    a.set_xlim(left=0)
    a.grid(alpha=0.3)
    a.legend(fontsize=10)
ax[1].set_ylim(bottom=0)
fig.tight_layout()
p = os.path.join(figs, f'FOUREL_expansion_{loading}{extra}{suffix}.png')
fig.savefig(p, dpi=150)
plt.close(fig)
show(p)

# ---------------------------------------------------------------- damage snapshots (exact pixels)
cmap_lab = ListedColormap(['#2b2b2b', '#9a9a9a', '#e8c547'])
for N in NS:
    keys = [k for k in [(N, snap_solver), (N, 'MS')] if k in runs]
    if not keys:
        continue
    targets = [0.0025, 0.005, 0.01, 0.02, 0.05]
    fig, ax = plt.subplots(len(keys), len(targets) + 1, figsize=(3.3 * (len(targets) + 1), 3.6 * len(keys)),
                           squeeze=False)
    for i, k in enumerate(keys):
        r = runs[k]
        st = r['steps']
        lab = st['label']
        ax[i, 0].imshow(lab.T, origin='lower', cmap=cmap_lab, interpolation='nearest', vmin=0, vmax=2)
        ax[i, 0].set_title('paste / aggregate / gel', fontsize=12)
        ax[i, 0].set_ylabel(STYLE[k[1]]['label'].replace(' (tol 1e-4)', ''), fontsize=12)
        for j, t in enumerate(targets):
            a = ax[i, j + 1]
            idx = int(np.argmin(np.abs(st['eps_asr'] - t)))
            if abs(st['eps_asr'][idx] - t) > 1e-6:
                a.axis('off')
                continue
            phi = np.load(os.path.join(r['base'] + '_damage', f'step_{idx:03d}.npy'))
            a.imshow(lab.T, origin='lower', cmap=cmap_lab, interpolation='nearest', vmin=0, vmax=2, alpha=0.35)
            a.imshow(np.ma.masked_less(phi.T, 0.05), origin='lower', cmap='Reds', interpolation='nearest', vmin=0,
                     vmax=1)
            a.set_title(f'$\\varepsilon_{{ASR}}$ = {100 * t:g} %', fontsize=12)
        for a in ax[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f'damage $\\varphi$ (red), {N}$^2$, loading {loading}', fontsize=13)
    fig.tight_layout()
    p = os.path.join(figs, f'FOUREL_damage_N{N}_{loading}{extra}{suffix}.png')
    fig.savefig(p, dpi=max(110, int(N / 3.0)))
    plt.close(fig)
    show(p)


# ---------------------------------------------------------------- cracks / fragments / percolation (their Fig. 6)
from scipy import ndimage


EIGHT = np.ones((3, 3), dtype=int)


def periodic_count(mask, structure=None):
    """number of clusters of mask on the periodic grid (4-connected, or 8 with structure=EIGHT; the periodic merge
    across the faces is 4-connected)"""
    lab, n = ndimage.label(mask, structure=structure)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in ((lab[0, :], lab[-1, :]), (lab[:, 0], lab[:, -1])):
        for p, q in zip(a, b):
            if p and q:
                parent[find(p)] = find(q)
    return len({find(i) for i in range(1, n + 1)})


def spans(mask):
    """an 8-connected damaged cluster connects opposite faces of the cell (non-periodic labelling)"""
    lab, n = ndimage.label(mask, structure=EIGHT)
    return bool((set(lab[0, :]) & set(lab[-1, :])) - {0} or (set(lab[:, 0]) & set(lab[:, -1])) - {0})


fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
perc = []
for k in sorted(runs, key=lambda k: (k[0], k[1])):
    if k[1] not in ('Green', 'MS', 'dual') or k[0] not in NS:
        continue
    r = runs[k]
    eps = r['steps']['eps_asr']
    nc, nf, pe = [], [], None
    for j in range(len(eps)):
        phi = np.load(os.path.join(r['base'] + '_damage', f'step_{j:03d}.npy'))
        nc.append(periodic_count(phi > 0.5, EIGHT))
        nf.append(periodic_count(phi <= 0.5))
        if pe is None and spans(phi > 0.5):
            pe = eps[j]
    perc.append(f'{k[0]} {k[1]}: first spanning crack at eps_ASR = '
                + ('-' if pe is None else f'{100 * pe:.3f} %') + f', max cracks {max(nc)}, final fragments {nf[-1]}')
    ls = {64: ':', 128: '--', 256: '-'}[k[0]]
    st = STYLE[k[1]]
    ax[0].plot(100 * eps, nc, ls=ls, color=st['color'], label=f'{k[0]}$^2$ {st["label"]}')
    ax[1].plot(100 * eps, nf, ls=ls, color=st['color'], label=f'{k[0]}$^2$ {st["label"]}')
for a, t in zip(ax, ('number of cracks ($\\varphi$ > 0.5, 8-connected)', 'number of fragments')):
    a.set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
    a.set_ylabel(t)
    a.set_xlim(left=0)
    a.set_ylim(bottom=0)
    a.grid(alpha=0.3)
ax[0].legend(fontsize=10)
fig.tight_layout()
p = os.path.join(figs, f'FOUREL_cracks_{loading}{extra}{suffix}.png')
fig.savefig(p, dpi=150)
plt.close(fig)
show(p)
print('\n'.join(perc))
with open(os.path.join(figs, f'FOUREL_cost_table_{loading}{extra}{suffix}.txt'), 'a') as f:
    f.write('\n'.join(perc) + '\n')
