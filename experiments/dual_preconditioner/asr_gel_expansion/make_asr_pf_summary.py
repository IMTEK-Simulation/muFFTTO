"""
One summary figure of the phase-field / Fourel study at 128^2:
  rows: old model (halos), A = Fourel-matched AT1, B = PF-CZM real concrete (vol. driving), B with Rankine driving,
        Fourel's nonlocal threshold model (sibling script fourel_nonlocal_damage.py, run FOUREL_N128_MS_free_k0.0001,
        read only), damage on the exact pixels at EPS_SHOW;
  bottom: macro volume change tr E, mean CG per elastic solve per load step for A and for B (Green / GJ / dual).
    python make_asr_pf_summary.py           env: EPS_SHOW (0.25,0.5,0.75,1.0,1.5), FOUREL_RUN
-> figures/ASRpf_summary_N128.png
"""
import os
import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data')
N = 128
eps_show = [float(e) / 100 for e in os.environ.get('EPS_SHOW', '0.25,0.5,0.75,1.0,1.5').split(',')]
fourel_run = os.environ.get('FOUREL_RUN', 'FOUREL_N128_MS_free_k0.0001')
plt.rcParams.update({'font.size': 13})
P_COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
P_NAME = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}


def load_asrpf(variant, p='dual', suf=''):
    base = os.path.join(data, f'ASRpf_{variant}_N{N}_{p}{suf}')
    if not os.path.exists(base + '_steps.npz'):
        return None
    r = dict(np.load(base + '_steps.npz', allow_pickle=True))
    key = 'loss' if glob.glob(base + '_damage/loss_*.npy') else 'step'
    r['field'] = [np.load(f) for f in sorted(glob.glob(base + f'_damage/{key}_*.npy'))][:len(r['eps_asr'])]
    r['trE'] = r['E'][:, 0] + r['E'][:, 1]
    return r


def load_longest(variant):
    """the run (any preconditioner: identical damage paths) that reached the largest load"""
    runs = [r for r in (load_asrpf(variant, p) for p in P_COLOR) if r is not None]
    return max(runs, key=lambda r: len(r['eps_asr'])) if runs else None


def load_fourel():
    base = os.path.join(data, fourel_run)
    if not os.path.exists(base + '_steps.npz'):
        return None
    r = dict(np.load(base + '_steps.npz', allow_pickle=True))
    n = min(len(r['eps_asr']), len(glob.glob(base + '_damage/step_*.npy')))
    r['field'] = [np.load(base + f'_damage/step_{k:03d}.npy') for k in range(n)]
    r['eps_asr'] = r['eps_asr'][:n]
    r['trE'] = (r['E'][:, 0] + r['E'][:, 1])[:n]
    return r


rows = [('old', 'old model\n(AT2, hybrid, Falsafi)', load_longest('old')),
        ('A', 'A: Fourel-matched\nAT1, Gc 0.53 J/m2', load_longest('fourel')),
        ('B', 'B: PF-CZM, Gf 60 J/m2\nvol. driving (1-omega)', load_longest('czm')),
        ('BR', 'B, Rankine driving\n(1-omega)', load_longest('czmR')),
        ('F', 'Fourel nonlocal\nthreshold (their model)', load_fourel())]
rows = [r for r in rows if r[2] is not None]
label = rows[0][2]['label']
agg = label >= 1
segs = [[(i + 1, j), (i + 1, j + 1)] for i in range(N - 1) for j in range(N) if agg[i, j] != agg[i + 1, j]]
segs += [[(i, j + 1), (i + 1, j + 1)] for i in range(N) for j in range(N - 1) if agg[i, j] != agg[i, j + 1]]

nr, nc = len(rows), len(eps_show)
fig = plt.figure(figsize=(3.3 * nc + 1.5, 3.3 * nr + 4.6))
gs = fig.add_gridspec(nr + 1, nc, height_ratios=[1] * nr + [1.25], left=0.13, right=0.9, top=0.97, bottom=0.05,
                      hspace=0.12, wspace=0.05)
for i, (key, title, r) in enumerate(rows):
    for j, e in enumerate(eps_show):
        ax = fig.add_subplot(gs[i, j])
        ax.set_xticks([])
        ax.set_yticks([])
        k = np.flatnonzero(np.abs(r['eps_asr'] - e) < 1e-9)
        if len(k) == 0:
            ax.text(0.5, 0.5, 'not reached', ha='center', va='center', transform=ax.transAxes)
        else:
            im = ax.imshow(r['field'][k[0]].T, origin='lower', interpolation='nearest', cmap='magma_r', vmin=0,
                           vmax=1, extent=(0, N, 0, N))
            ax.add_collection(LineCollection(segs, colors='tab:cyan', linewidths=0.35))
        if i == 0:
            ax.set_title(f'eps_ASR = {100 * e:g} %')
        if j == 0:
            ax.set_ylabel(title, fontsize=12)
pos = fig.axes[nc - 1].get_position()
cax = fig.add_axes([pos.x1 + 0.012, 0.45, 0.012, 0.35])
fig.colorbar(im, cax=cax, label='damage d (PF-CZM: 1 - omega(d))')

sub = gs[nr, :].subgridspec(1, 3, wspace=0.38)
ax = fig.add_subplot(sub[0])
styles = {'old': ('tab:gray', '-'), 'A': ('k', '-'), 'B': ('tab:purple', '--'), 'BR': ('tab:orange', '-.'),
          'F': ('tab:brown', ':')}
for key, title, r in rows:
    ax.plot(100 * r['eps_asr'], 100 * r['trE'], color=styles[key][0], ls=styles[key][1], lw=2,
            label=title.split('\n')[0])
ax.set_xlabel('gel eigenstrain eps_ASR [%]')
ax.set_ylabel('macro volume change tr E [%]')
ax.set_xlim(0, 100 * max(eps_show) + 0.25)
ax.grid(alpha=0.3)
ax.legend(fontsize=10, loc='upper left')
for col, (variant, name) in enumerate((('fourel', 'A'), ('czm', 'B'))):
    ax = fig.add_subplot(sub[col + 1])
    for p in P_COLOR:
        r = load_asrpf(variant, p)
        if r is None:
            continue
        ax.plot(100 * r['eps_asr'], r['cg_main_mean'], color=P_COLOR[p], lw=1.8, marker='o', ms=3, label=P_NAME[p])
    ax.set_xlabel('gel eigenstrain eps_ASR [%]')
    ax.set_ylabel('mean CG per elastic solve')
    ax.set_title(f'{name}: CG per elastic solve, {N}^2', fontsize=13)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
out = os.path.join(here, 'figures', f'ASRpf_summary_N{N}.png')
fig.savefig(out, dpi=100)
print('->', out)
