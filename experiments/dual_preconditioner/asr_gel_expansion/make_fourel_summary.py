"""
One summary figure of the Fourel-model solver comparison (fourel_nonlocal_damage.py runs, LOADING=free, k = 1e-4):
* top: damage at 128^2 and 256^2, MS basic scheme (tol 1e-6, equal accuracy) vs Newton + the cheapest preconditioner,
  three eps_ASR each (exact pixels)
* bottom left: iterations per elastic solve vs eps_ASR at 128^2 (all solvers)
* bottom right: total cost per grid up to a common eps_ASR, in Green-iteration units (dual = 2), with the median
  measured energy error of each solver
-> figures/FOUREL_summary.png (opened with xdg-open)

    python make_fourel_summary.py      # env NO_OPEN=1 to skip xdg-open
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data')
plt.rcParams.update({'font.size': 13})
COST = {'MS': 1.0, 'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}
KEYS = [('MS', ''), ('MS', '_tolms1e-06'), ('Green', ''), ('Green_Jacobi', ''), ('dual', '')]
STYLE = {('MS', ''): dict(color='black', ls='--', label='MS basic scheme, tol 1e-4'),
         ('MS', '_tolms1e-06'): dict(color='grey', ls='-', label='MS basic scheme, tol 1e-6'),
         ('Green', ''): dict(color='tab:green', ls='-', label='Newton + Green'),
         ('Green_Jacobi', ''): dict(color='tab:blue', ls='-', label='Newton + Green-Jacobi'),
         ('dual', ''): dict(color='tab:red', ls='-', label='Newton + dual')}


def load(N, s, tag):
    base = os.path.join(data, f'FOUREL_N{N}_{s}_free_k0.0001{tag}')
    if not os.path.exists(base + '_steps.npz'):
        return None
    st = dict(np.load(base + '_steps.npz', allow_pickle=True))
    sv = np.atleast_1d(np.genfromtxt(base + '_solves.csv', delimiter=',', names=True, dtype=None, encoding=None))
    sv = sv[sv['step'] < len(st['eps_asr'])]
    return dict(st=st, sv=sv, base=base)


def cost(key, r, e):
    sv, st = r['sv'], r['st']
    return COST[key[0]] * (sv['linear_its'][sv['eps_asr'] <= e + 1e-12].sum()
                           + st['unit_its'][st['eps_asr'] <= e + 1e-12].sum())


runs = {(N,) + k: load(N, *k) for N in (64, 128, 256) for k in KEYS}
runs = {k: v for k, v in runs.items() if v is not None}
common = {N: min(r['st']['eps_asr'][-1] for k, r in runs.items() if k[0] == N) for N in (64, 128, 256)
          if any(k[0] == N for k in runs)}
best = {}
for N in common:
    c = {k[1]: cost(k[1:], r, common[N]) for k, r in runs.items() if k[0] == N and k[1] != 'MS'}
    best[N] = min(c, key=c.get)
print('common eps_ASR:', common, 'cheapest preconditioner:', best)

nrows = sum(2 for N in (128, 256) if N in common)
fig = plt.figure(figsize=(16, 3.9 * nrows + 6.5))
frac = 3.9 * nrows / (3.9 * nrows + 6.5)
gs = GridSpec(nrows, 4, figure=fig, left=0.08, right=0.98, top=0.97, bottom=1 - frac + 0.02, hspace=0.22,
              wspace=0.05)
gb = GridSpec(1, 2, figure=fig, left=0.08, right=0.98, top=1 - frac - 0.03, bottom=0.1, wspace=0.28)
cmap_lab = ListedColormap(['#2b2b2b', '#9a9a9a', '#e8c547'])
row = 0
for N in (128, 256):
    if N not in common:
        continue
    for key in [(N, 'MS', '_tolms1e-06'), (N, best[N], '')]:
        if key not in runs:
            continue
        r = runs[key]
        st = r['st']
        lab = st['label']
        eps = st['eps_asr']
        targets = [common[N] * f for f in (0.4, 0.7, 1.0)]
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(lab.T, origin='lower', cmap=cmap_lab, interpolation='nearest', vmin=0, vmax=2)
        ax.set_ylabel(f'{N}$^2$\n' + STYLE[key[1:]]['label'].replace(', ', '\n'), fontsize=12)
        if row == 0:
            ax.set_title('paste / aggregate / gel', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for j, t in enumerate(targets):
            idx = int(np.argmin(np.abs(eps - t)))
            phi = np.load(os.path.join(r['base'] + '_damage', f'step_{idx:03d}.npy'))
            a = fig.add_subplot(gs[row, j + 1])
            a.imshow(lab.T, origin='lower', cmap=cmap_lab, interpolation='nearest', vmin=0, vmax=2, alpha=0.35)
            a.imshow(np.ma.masked_less(phi.T, 0.05), origin='lower', cmap='Reds', interpolation='nearest', vmin=0,
                     vmax=1)
            a.set_title(f'$\\varepsilon_{{ASR}}$ = {100 * eps[idx]:.3g} %', fontsize=12)
            a.set_xticks([])
            a.set_yticks([])
        row += 1

ax1 = fig.add_subplot(gb[0, 0])
for key in KEYS:
    k = (128,) + key
    if k not in runs:
        continue
    sv = runs[k]['sv']
    steps = np.unique(sv['step']).astype(int)
    e = np.array([sv['eps_asr'][sv['step'] == s][0] for s in steps])
    its = np.array([sv['linear_its'][sv['step'] == s].mean() for s in steps]) * COST[key[0]]
    ax1.plot(100 * e, its, color=STYLE[key]['color'], ls=STYLE[key]['ls'], label=STYLE[key]['label'])
ax1.set_xlabel(r'gel eigenstrain $\varepsilon_{ASR}$ [%]')
ax1.set_ylabel('cost per elastic solve\n[Green-iteration units]')
ax1.set_title('128$^2$: one solve per crack spot')
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gb[0, 1])
Ns = sorted(common)
w = 0.8 / len(KEYS)
for j, key in enumerate(KEYS):
    for i, N in enumerate(Ns):
        k = (N,) + key
        if k not in runs:
            continue
        c = cost(key, runs[k], common[N])
        ref = cost(('Green', ''), runs[(N, 'Green', '')], common[N]) if (N, 'Green', '') in runs else c
        x = i + (j - (len(KEYS) - 1) / 2) * w
        ax2.bar(x, c / ref, w, color=STYLE[key]['color'], hatch='//' if key == ('MS', '') else None,
                edgecolor='white' if key != ('MS', '') else 'grey', label=STYLE[key]['label'] if i == 0 else None)
        err = runs[k]['sv']['energy_error']
        err = err[(~np.isnan(err)) & (runs[k]['sv']['eps_asr'] <= common[N] + 1e-12)]
        if err.size:
            ax2.text(x, c / ref, f'{np.median(err):.0e}', ha='center', va='bottom', fontsize=9, rotation=90)
ax2.set_xticks(range(len(Ns)))
ax2.set_xticklabels([f'{N}$^2$\nto {100 * common[N]:.3g} %' for N in Ns])
ax2.set_ylabel('total cost / Newton + Green')
ax2.set_title('total cost (labels: median energy error)')
ax2.set_ylim(0, ax2.get_ylim()[1] * 1.25)
ax2.grid(axis='y', alpha=0.3)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=STYLE[k]['color'], ls=STYLE[k]['ls'], lw=3) for k in KEYS],
           [STYLE[k]['label'] for k in KEYS], loc='lower center', ncol=3, frameon=False, fontsize=12)
fig.suptitle('Fourel et al. nonlocal ASR damage (2D, free, one spot per solve): MS basic scheme vs Newton-PCG',
             fontsize=15, y=0.995)
out = os.path.join(here, 'figures', 'FOUREL_summary.png')
fig.savefig(out, dpi=130)
print('->', out)
if os.environ.get('NO_OPEN', '0') != '1':
    os.system(f'xdg-open {out} >/dev/null 2>&1 &')
