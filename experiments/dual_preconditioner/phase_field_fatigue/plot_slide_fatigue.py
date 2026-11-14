"""Slide figures of the NEMO fatigue mesh study (style of ../topology_optimization/plot_slide_green_grid_independence.py).

    python plot_slide_fatigue.py
    -> figures/fatigue_cg_vs_grid_2D.{png,pdf}   CG per elastic solve during crack growth vs grid, 2D 64^2 ... 1024^2
       figures/fatigue_cg_vs_grid_3D.{png,pdf}   the same, 3D 32^3 ... 256^3 (runs not finished: open marker, dashed)
       figures/fatigue_walltime_1024.{png,pdf}   wall time of the 2D 1024^2 runs (Elapsed time in the logs)
Grey band: l/h < 2 (under-resolved crack band). "Growth" = from the first cycle the crack is > 2.5 pixels longer than
the pre-crack to the final break (same definition as plot_FAT_mesh_study.py).
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
         ('dual', r'dual ($\times 2$)', 'tab:red', '^')]            # dual iterations counted x2 (cost)
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})


def load(D, N, P):
    files = glob.glob(os.path.join(HERE, 'exp_data', f'FAT_N{N}_{P}_incl*_nemo{D}*_precrack_*_cycles.csv'))
    files = [f for f in files if re.search(rf'_nemo{D}(_r\d+)?_precrack', f)]
    if not files:
        return None
    files.sort(key=lambda f: int(m[1]) if (m := re.search(r'_nemo\dD_r(\d+)_precrack', f)) else 0)
    out = None
    for f in files:
        p = np.genfromtxt(f, delimiter=',', names=True, ndmin=1)
        if p.size:
            out = p if out is None else np.concatenate([out[out['step'] < p['step'].min()], p])
    return out


def growth_cg(D, N, P):
    d = load(D, N, P)
    if d is None:
        return None
    grown = d['crack_length'] > d['crack_length'][0] + 2.5 / N
    i0 = int(np.argmax(grown)) if grown.any() else len(d)
    broken = np.r_[False, d['sigma'][1:] < 0.7 * d['sigma'][:-1]]
    i1 = int(np.argmax(broken)) if broken.any() else len(d)
    s = slice(i0, i1)
    logs = glob.glob(os.path.join(HERE, 'logs', f'FAT_nemo{D}_N{N}_{P}.log')) + glob.glob(
        os.path.join(HERE, 'logs', f'FAT_nemo{D}_N{N}_{P}_r[0-9]*.log'))    # finished = a segment's log has 'Elapsed time'
    done = any('Elapsed time' in open(log).read() for log in logs)
    return d['cg_u_step'][s].sum() / d['staggered_its'][s].sum(), done


def cg_plot(D, grids, fname, ylim):
    fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
    sup = '^2' if D == '2D' else '^3'
    for P, label, c, m in PRECS:
        pts = [(N, *r) for N in grids if (r := growth_cg(D, N, P)) is not None]
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts]) * (2 if P == 'dual' else 1)
        done = np.array([p[2] for p in pts])
        nd = len(x) if done.all() else int(np.argmin(done))                # finished runs first
        ax.plot(x[:nd], y[:nd], '-', color=c, lw=2.2, clip_on=False, label=label)
        if nd < len(x):
            ax.plot(x[max(nd - 1, 0):], y[max(nd - 1, 0):], '--', color=c, lw=2.2, clip_on=False)
        for xi, yi, di in zip(x, y, done):
            ax.plot(xi, yi, m, color=c, ms=6.5, mfc=c if di else 'white', mew=1.6, clip_on=False, zorder=5)
            if not di:
                ax.annotate('running', (xi, yi), xytext=(0, -9), textcoords='offset points', ha='center',
                            va='top', fontsize=10, color=c)
    if D == '3D':       # 256^3 still running: final value estimated from the running mean and the growth of the finished 128^3 runs
        for v, c in ((280, 'tab:green'), (2 * 117, 'tab:red')):
            ax.plot(256, v, '*', color=c, ms=13, mec='k', mew=0.6, clip_on=False, zorder=6)
        ax.plot([], [], '*', color='0.5', ms=13, mec='k', mew=0.6, ls='none', label='estimated final')
    ax.axvspan(grids[0] / 1.5, 128, color='0.9', zorder=0)
    ax.text(np.sqrt(grids[0] * 128), ylim * (0.3 if D == '2D' else 0.03), '$\\ell/h < 2$', ha='center', va='bottom', fontsize=11, color='0.4')
    ax.set_xscale('log', base=2)
    ax.set_xticks(grids), ax.set_xticklabels([f'${n}{sup}$' for n in grids]), ax.minorticks_off()
    ax.set_xlim(grids[0], grids[-1])
    ax.set_ylim(0, ylim)
    ax.set(xlabel='grid size', ylabel=r'CG per elastic solve (dual $\times 2$)')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right' if D == '3D' else 'upper left', frameon=False, fontsize=12)
    out = os.path.join(HERE, 'figures', fname)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
    print('wrote', out)
    plt.close(fig)


cg_plot('2D', [64, 128, 256, 512, 1024], 'fatigue_cg_vs_grid_2D', 1000)
cg_plot('3D', [32, 64, 128, 256], 'fatigue_cg_vs_grid_3D', 520)

# wall time 1024^2 from the logs ("Elapsed time: ... s")
T = {}
for P, _, _, _ in PRECS:
    txt = open(os.path.join(HERE, 'logs', f'FAT_nemo2D_N1024_{P}.log')).read()
    T[P] = float(re.findall(r'Elapsed time: ([\d.]+) s', txt)[-1]) / 3600
fig, ax = plt.subplots(figsize=(4.4, 3.3), layout='constrained')
for i, (P, label, c, _) in enumerate(PRECS):
    ax.bar(i, T[P], color=c, width=0.62)
    ax.text(i, T[P] + 0.25, f'{T[P]:.1f} h', ha='center', va='bottom', fontsize=13)
ax.set_xticks(range(3)), ax.set_xticklabels([p[1] for p in PRECS])
ax.set_ylim(0, 14)
ax.set_ylabel('wall time [h]')
ax.set_title('$1024^2$, 174 cycles, one L40S GPU', fontsize=13)
ax.grid(alpha=0.3, axis='y')
out = os.path.join(HERE, 'figures', 'fatigue_walltime_1024')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200)
print('wrote', out, {k: round(v, 2) for k, v in T.items()})
