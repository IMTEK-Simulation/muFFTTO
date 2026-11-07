"""
Figures and tables of asr_pf_fourel.py runs (exp_data/ASRpf_*).

    python plot_asr_pf_fourel.py
env: N (128), SUFFIX (''), VARIANTS (old,hybrid,at2,amor,softgel,fourel), EPS_SHOW (0.25,0.5,0.75,1.0,1.5 [%]),
     PRECOND_VARIANT (fourel), PRECOND_GRIDS (64,128,256)

* figures/ASRpf_damage_variants_N<N>.png  damage (exact pixels) per variant (dual run) at EPS_SHOW
* figures/ASRpf_response_N<N>.png         macro volume change tr E, stiffness ratio, cracks, fragments vs eps_ASR
* figures/ASRpf_cg_<variant>.png          CG per elastic solve per load step and cumulative CG, G / GJ / dual
* stdout: crack initiation / percolation table and the CG table (before / during cracking, totals, dual x2, wall time)
Cracks = periodic 4-connected clusters of d > 0.5, fragments = clusters of d <= 0.5 (Fourel Sec. 3.4.2); percolation =
the first step at which a crack cluster wraps around the periodic cell.
"""
import os
import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data')
figs = os.path.join(here, 'figures')
N = int(os.environ.get('N', 128))
suffix = os.environ.get('SUFFIX', '')
variants = os.environ.get('VARIANTS', 'old,hybrid,at2,amor,softgel,fourel').split(',')
eps_show = [float(e) / 100 for e in os.environ.get('EPS_SHOW', '0.25,0.5,0.75,1.0,1.5').split(',')]
pre_variant = os.environ.get('PRECOND_VARIANT', 'fourel')
pre_grids = [int(g) for g in os.environ.get('PRECOND_GRIDS', '64,128,256').split(',')]
TITLES = {'old': 'old model (Falsafi mat., AT2, hybrid)', 'czm_b1': 'B: PF-CZM, b = 1 mm', 'hybrid': 'Fourel, no stress split',
          'at2': 'Fourel, AT2 (no threshold)', 'amor': 'Fourel, Amor psi+ (vol+dev)',
          'softgel': 'Fourel, soft undamaged gel', 'fourel': 'Fourel set-up (AT1, split, vol psi+)',
          'fourel_k1e-5': 'Fourel, k = 1e-5', 'czm': 'B: PF-CZM real concrete (1-omega)'}
P_COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
P_NAME = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
plt.rcParams.update({'font.size': 13})


def periodic_clusters(mask):
    """number of periodic 4-connected clusters of mask, and whether one wraps around the torus"""
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, False
    edges = []
    for axis in (0, 1):
        a = np.take(lab, -1, axis=axis)
        b = np.take(lab, 0, axis=axis)
        for la, lb in set(zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)])):
            shift = np.zeros(2, dtype=int)
            shift[axis] = 1
            edges.append((la, lb, shift))
    adj = {i: [] for i in range(1, n + 1)}
    for la, lb, s in edges:
        adj[la].append((lb, s))
        adj[lb].append((la, -s))
    pos, nb_clusters, wraps = {}, 0, False
    for start in range(1, n + 1):
        if start in pos:
            continue
        nb_clusters += 1
        pos[start] = np.zeros(2, dtype=int)
        stack = [start]
        while stack:
            c = stack.pop()
            for nb, s in adj[c]:
                if nb not in pos:
                    pos[nb] = pos[c] + s
                    stack.append(nb)
                elif np.any(pos[nb] != pos[c] + s):
                    wraps = True
    return nb_clusters, wraps


def run_files(variant, n, p, suf=suffix):
    base = os.path.join(data, f'ASRpf_{variant}_N{n}_{p}{suf}')
    return base + '_steps.npz', base + '_damage', base + '_iterations.csv'


def load_run(variant, n, p, suf=suffix):
    if '_' in variant:                    # 'czm_b1' -> variant czm, run suffix _b1 (fourel_k1e-5 likewise)
        variant, extra = variant.split('_', 1)
        suf = '_' + extra + suf
    sf, dd, itf = run_files(variant, n, p, suf)
    if not os.path.exists(sf):
        return None
    r = dict(np.load(sf, allow_pickle=True))
    r['damage'] = [np.load(f) for f in sorted(glob.glob(os.path.join(dd, 'step_*.npy')))][:len(r['eps_asr'])]
    loss = sorted(glob.glob(os.path.join(dd, 'loss_*.npy')))       # PF-CZM: stiffness loss 1 - omega(d)
    if loss:
        r['damage'] = [np.load(f) for f in loss][:len(r['eps_asr'])]
    r['it'] = np.genfromtxt(itf, delimiter=',', names=True)
    cr, fr, wr = [], [], []
    for d in r['damage']:
        dp = d[:n, :n] if d.shape[0] >= n else d
        nc, w = periodic_clusters(dp > 0.5)
        nf, _ = periodic_clusters(dp <= 0.5)
        cr.append(nc)
        fr.append(nf)
        wr.append(w)
    r['cracks'], r['fragments'], r['wraps'] = np.array(cr), np.array(fr), np.array(wr)
    r['field_max'] = np.array([float(d.max()) for d in r['damage']])   # d (AT1/AT2) or 1 - omega (PF-CZM)
    return r


def crack_metrics(r):
    eps = r['eps_asr']
    i_init = np.flatnonzero(r['field_max'] > 0.5)
    i_perc = np.flatnonzero(r['wraps'])
    return (eps[i_init[0]] if len(i_init) else np.nan, eps[i_perc[0]] if len(i_perc) else np.nan)


def aggregate_outline(ax, label):
    """aggregate boundaries on the pixel edges (no smoothing)"""
    n = label.shape[0]
    agg = label >= 1
    segs = []
    for i in range(n):
        for j in range(n):
            if agg[i, j] != agg[(i + 1) % n, j] and i + 1 < n:
                segs.append([(i + 1, j), (i + 1, j + 1)])
            if agg[i, j] != agg[i, (j + 1) % n] and j + 1 < n:
                segs.append([(i, j + 1), (i + 1, j + 1)])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors='tab:cyan', linewidths=0.5))


# ------------------------------------------------------------------ variants: damage maps + response (dual runs)
runs = {v: load_run(v, N, 'dual') for v in variants}
runs = {v: r for v, r in runs.items() if r is not None}
os.makedirs(figs, exist_ok=True)
if runs:
    print(f'\nvariants at {N}^2 (dual): crack initiation (max d > 0.5) / percolation (crack cluster wraps the cell), '
          f'eps_ASR [%]; final tr E [%], stiffness ratio, cracks, fragments, steps, unconverged steps')
    for v, r in runs.items():
        ei, ep = crack_metrics(r)
        print(f'  {v:10s} init {100 * ei:.3f}  perc {100 * ep:.3f}  final eps {100 * r["eps_asr"][-1]:.3f}: '
              f'tr E {100 * (r["E"][-1, 0] + r["E"][-1, 1]):.4f}, |C|/|C0| {r["stiffness_ratio"][-1]:.3f}, '
              f'cracks {r["cracks"][-1]}, fragments {r["fragments"][-1]}, steps {len(r["eps_asr"])}, '
              f'unconverged {int(np.sum(1 - r["converged"]))}')
    nr, nc = len(runs), len(eps_show) + 1
    fig, axes = plt.subplots(nr, nc, figsize=(2.9 * nc, 3.0 * nr + 0.6), squeeze=False)
    for row, (v, r) in enumerate(runs.items()):
        lab = r['label']
        ax = axes[row, 0]
        ax.imshow(lab.T, origin='lower', interpolation='nearest', cmap=matplotlib.colors.ListedColormap(
            ['#d9d9d9', '#8c8c8c', '#e6550d']), vmin=-0.5, vmax=2.5, extent=(0, N, 0, N))
        ax.set_ylabel(TITLES.get(v, v).replace(', ', ',\n', 1).replace(' real', '\nreal'), fontsize=12)
        if row == 0:
            ax.set_title('phases', fontsize=13)
        for col, e in enumerate(eps_show):
            ax = axes[row, col + 1]
            k = int(np.argmin(np.abs(r['eps_asr'] - e)))
            if abs(r['eps_asr'][k] - e) > 1e-9:
                ax.axis('off')
                continue
            im = ax.imshow(r['damage'][k].T, origin='lower', interpolation='nearest', cmap='magma_r', vmin=0, vmax=1,
                           extent=(0, N, 0, N))
            aggregate_outline(ax, lab)
            if row == 0:
                ax.set_title(f'eps_ASR = {100 * e:g} %', fontsize=13)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(left=0.08, right=0.92, top=0.95, bottom=0.02, wspace=0.05, hspace=0.08)
    pos = axes[0, -1].get_position()
    cax = fig.add_axes([pos.x1 + 0.01, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label='damage d  (PF-CZM: 1 - omega(d))')
    f = os.path.join(figs, f'ASRpf_damage_variants_N{N}{suffix}.png')
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print('->', f)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]
    cmap = plt.get_cmap('tab10')
    for i, (v, r) in enumerate(runs.items()):
        kw = dict(color='k' if v == 'fourel' else cmap(i), ls=styles[i % len(styles)], lw=2.2 if v == 'fourel' else 1.6,
                  label=TITLES.get(v, v))
        e = 100 * r['eps_asr']
        axes[0, 0].plot(e, 100 * (r['E'][:, 0] + r['E'][:, 1]), **kw)
        axes[0, 1].plot(e, r['stiffness_ratio'], **kw)
        axes[1, 0].plot(e, r['cracks'], **kw)
        axes[1, 1].plot(e, r['fragments'], **kw)
    for ax, yl in zip(axes.ravel(), ('macro volume change tr E [%]', 'stiffness ratio |C_eff| / |C_eff,0|',
                                     'number of cracks (d > 0.5)', 'number of fragments (d <= 0.5)')):
        ax.set_xlabel('gel eigenstrain eps_ASR [%]')
        ax.set_ylabel(yl)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.3)
    axes[0, 1].set_ylim(0, 1.05)
    h, l_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc='upper center', ncol=3, fontsize=12, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    f = os.path.join(figs, f'ASRpf_response_N{N}{suffix}.png')
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print('->', f)

# ------------------------------------------------------------------ preconditioners
print(f'\npreconditioners, variant {pre_variant}: mean CG per main elastic solve in load steps ending with max d < 0.9 / >= 0.9 (PF-CZM: 1-omega); '
      f'cracking; total elastic CG (main + unit solves); dual x2 = cost in Green-iteration units (2D); wall time')
grid_runs = {}
for n in pre_grids:
    for p in P_COLOR:
        r = load_run(pre_variant, n, p)
        if r is not None:
            grid_runs[(n, p)] = r
lines = []
for n in pre_grids:
    nsteps = [len(grid_runs[(n, p)]['eps_asr']) for p in P_COLOR if (n, p) in grid_runs]
    n_common = min(nsteps) if nsteps else 0      # compare over the load steps all preconditioners have
    for p in P_COLOR:
        r = grid_runs.get((n, p))
        if r is None:
            continue
        it = r['it'][r['it']['step'] < n_common]
        pre = r['field_max'][it['step'].astype(int)] < 0.9     # load steps ending with max d (1-omega) < 0.9
        tot = int(np.sum(r['cg_all_sum'][:n_common]))
        cost = 2 * tot if p == 'dual' else tot
        wall = float(np.sum(r['time_step'][:n_common]))
        line = (f'  {n:4d}^2 {P_NAME[p]:13s} steps {n_common:3d} (eps <= {100 * r["eps_asr"][n_common - 1]:.3g} %)  '
                f'its {len(it):5d}  '
                f'CG/solve {np.mean(it["cg_main"][pre]) if pre.any() else np.nan:6.1f} / '
                f'{np.mean(it["cg_main"][~pre]) if (~pre).any() else np.nan:6.1f}  total CG {tot:8d}  '
                f'cost (x2 dual) {cost:8d}  wall {wall:7.0f} s')
        lines.append(line)
        print(line)
if grid_runs:
    ns = sorted({n for n, _ in grid_runs})
    fig, axes = plt.subplots(2, len(ns), figsize=(5.2 * len(ns), 8.5), squeeze=False)
    for c, n in enumerate(ns):
        for p in P_COLOR:
            r = grid_runs.get((n, p))
            if r is None:
                continue
            e = 100 * r['eps_asr']
            axes[0, c].plot(e, r['cg_main_mean'], color=P_COLOR[p], marker='o', ms=3, label=P_NAME[p])
            axes[1, c].plot(e, np.cumsum(r['cg_all_sum']) * (2 if p == 'dual' else 1), color=P_COLOR[p],
                            label=P_NAME[p] + (' (x2)' if p == 'dual' else ''))
        axes[0, c].set_title(f'{n}^2 (l/h = {0.5 / (35.0 / n):.2g})')
        axes[0, c].set_ylabel('mean CG per elastic solve')
        axes[1, c].set_ylabel('cumulative elastic CG (dual x2)')
        for ax in axes[:, c]:
            ax.set_xlabel('gel eigenstrain eps_ASR [%]')
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
    h, l_ = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l_, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    f = os.path.join(figs, f'ASRpf_cg_{pre_variant}{suffix}.png')
    fig.savefig(f, dpi=110)
    plt.close(fig)
    print('->', f)
