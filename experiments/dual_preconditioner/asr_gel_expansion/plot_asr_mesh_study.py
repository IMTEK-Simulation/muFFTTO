"""
Mesh-size study figures (NEMO runs of asr_AT2_mesh_study.py, l = 1.5625 mm fixed, cheap zero-stress control):
  figures/asrmesh_<D>D_cg_vs_lh.png         mean CG per elastic solve vs l/h, before cracking / during cracking
  figures/asrmesh_<D>D_cg_vs_load.png       CG per elastic solve vs dV/V, one panel per grid
  figures/asrmesh_<D>D_stiffness.png        ||C||/||C0|| vs dV/V per grid (+ Falsafi et al. Fig. 8a in 2D)
  figures/asrmesh_2D_damage.png             damage at dV/V = 0.016 / 0.032 / 0.064 per grid (dual runs)
Grids with l/h < 2 are under-resolved: open markers / thin lines, '(under-resolved)' in the labels.
    python plot_asr_mesh_study.py [2|3]
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
dims = [int(sys.argv[1])] if len(sys.argv) > 1 else [2, 3]
L = 0.015625
PREC = {'Green': ('Green', 'tab:green', 'o', '-'), 'Green_Jacobi': ('Green-Jacobi', 'tab:blue', 's', '--'),
        'dual': ('dual', 'tab:red', '^', '-.')}
D_CRACK = 0.9
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 11})
fig_dir = os.path.join(here, 'figures')


def stem(D, N, P):
    return os.path.join(here, 'exp_data', f'ASRmesh_{D}D_N{N}_{P}_l{L:g}_cheap')


def load_it(D, N, P):
    f = stem(D, N, P) + '_iterations.csv'
    if not os.path.exists(f) or os.path.getsize(f) < 200:
        return None
    return np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None))


def save(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('saved', path)


for D in dims:
    grids = sorted({int(os.path.basename(f).split('_N')[1].split('_')[0])
                    for f in glob.glob(os.path.join(here, 'exp_data', f'ASRmesh_{D}D_N*_l{L:g}_cheap_iterations.csv'))})
    if not grids:
        continue
    data = {(N, P): load_it(D, N, P) for N in grids for P in PREC}
    lh = {N: L * N for N in grids}

    def tag(N):
        return f'{N}$^{D}$ (l/h = {lh[N]:g}{", under-resolved" if lh[N] < 2 else ""})'

    # ---- CG vs l/h
    table = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for P, (lab, col, mk, ls) in PREC.items():
        xs, pre, post = [], [], []
        for N in grids:
            it = data[(N, P)]
            if it is None:
                continue
            others = [data[(N, Q)] for Q in PREC if data[(N, Q)] is not None]
            last = min(int(o['step'].max()) for o in others)     # load steps done by all runs of this grid
            it = it[it['step'] <= last]
            cracked = it['max_d'] >= D_CRACK
            xs.append(lh[N])
            pre.append(it['cg_main'][~cracked].mean() if (~cracked).any() else np.nan)
            post.append(it['cg_main'][cracked].mean() if cracked.any() else np.nan)
            table.append((P, N, lh[N], last + 1, int(it['step'].max()) + 1, pre[-1], post[-1], len(it),
                          it['time_u'].sum() + it['time_d'].sum()))
        xs = np.array(xs)
        for ax, vals in zip(axes, (np.array(pre), np.array(post))):
            ax.plot(xs, vals, color=col, ls=ls, lw=1.5, label=lab)
            res = xs >= 2
            ax.plot(xs[res], vals[res], color=col, marker=mk, ls='none', ms=8)
            ax.plot(xs[~res], vals[~res], color=col, marker=mk, ls='none', ms=8, mfc='white')
    for ax, title in zip(axes, (f'before cracking (max d < {D_CRACK})', f'cracking (max d $\\geq$ {D_CRACK})')):
        ax.set_xscale('log', base=2)
        ax.set_xticks([lh[N] for N in grids])
        ax.set_xticklabels([f'{lh[N]:g}\n({N}$^{D}$)' for N in grids])
        ax.minorticks_off()
        ax.set_ylim(0, None)
        ax.set_xlabel('l / h  (grid)')
        ax.set_ylabel('mean CG iterations per elastic solve')
        ax.set_title(f'{D}D, {title}')
        ax.axvspan(min(lh.values()) / 1.5, 1.9, color='0.9', zorder=0)
        ax.grid(alpha=0.3)
    axes[0].legend(title='open / grey: l/h < 2')
    save(fig, f'asrmesh_{D}D_cg_vs_lh.png')
    print(f'{D}D: precond, N, l/h, common steps, own steps, CG pre, CG crack, staggered its, time [s]')
    for row in table:
        print('  {:13s} {:5d} {:5g} {:4d} {:4d} {:8.1f} {:8.1f} {:6d} {:9.0f}'.format(*row))

    # ---- CG vs load, one panel per grid
    fig, axes = plt.subplots(1, len(grids), figsize=(4.6 * len(grids), 4.2), squeeze=False)
    for ax, N in zip(axes[0], grids):
        for P, (lab, col, mk, ls) in PREC.items():
            it = data[(N, P)]
            if it is None:
                continue
            steps = np.unique(it['step'])
            ax.plot([100 * it['dVV'][it['step'] == s][0] for s in steps],
                    [it['cg_main'][it['step'] == s].mean() for s in steps], color=col, ls=ls, label=lab)
        ax.set_title(tag(N), fontsize=12)
        ax.set_xlabel(r'$\Delta V/V$ [%]')
        ax.set_ylabel('CG per elastic solve')
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        ax.grid(alpha=0.3)
    axes[0, 0].legend()
    save(fig, f'asrmesh_{D}D_cg_vs_load.png')

    # ---- stiffness reduction
    fig, ax = plt.subplots(figsize=(7.5, 5))
    shades = np.linspace(0.75, 0.0, len(grids))
    for N, sh in zip(grids, shades):
        for P in ('dual', 'Green', 'Green_Jacobi'):
            f = stem(D, N, P) + '_steps.npz'
            if os.path.exists(f):
                st = np.load(f)
                ax.plot(np.r_[0, 100 * st['dVV']], np.r_[1, st['stiffness_ratio']], color=str(sh),
                        lw=2.2 if lh[N] >= 2 else 1.0, ls='-' if lh[N] >= 2 else ':', label=tag(N) + f' [{P}]')
                break
    if D == 2:
        xs = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 6.4]
        ax.plot(xs, [1, 0.93, 0.84, 0.74, 0.645, 0.52, 0.42, 0.35, 0.30, 0.285], color='tab:orange', marker='o',
                ms=4, label='Falsafi et al. Fig. 8a (crack band, 128$^2$)')
    ax.set_xlabel(r'gel expansion $\Delta V/V$ [%]')
    ax.set_ylabel(r'$\|C_\mathrm{eff}\| / \|C_\mathrm{eff,0}\|$')
    ax.set_xlim(0, 6.6)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f'{D}D, l = 1.5625 mm')
    save(fig, f'asrmesh_{D}D_stiffness.png')

    # ---- damage snapshots (dual): 2D full fields, 3D mid-plane
    levels = [(15, 0.016), (31, 0.032), (63, 0.064)]
    rows = [N for N in grids if os.path.isdir(stem(D, N, 'dual') + '_damage')]
    if rows:
        fig, axes = plt.subplots(len(rows), len(levels), figsize=(4.2 * len(levels), 4.0 * len(rows)),
                                 squeeze=False, layout='constrained')
        for r, N in enumerate(rows):
            st = np.load(stem(D, N, 'dual') + '_steps.npz')
            lab = st['label']
            for c, (k, lev) in enumerate(levels):
                ax = axes[r, c]
                f = os.path.join(stem(D, N, 'dual') + '_damage', f'step_{k:03d}' + ('_midz' if D == 3 else '') + '.npy')
                if not os.path.exists(f):
                    ax.set_axis_off()
                    continue
                im = ax.imshow(np.clip(np.load(f), 0, 1).T, origin='lower', cmap='inferno', vmin=0, vmax=1,
                               interpolation='nearest', extent=(0, 100, 0, 100))
                cen = (np.arange(N) + 0.5) / N * 100
                ax.contour(cen, cen, (lab == 1).T.astype(float), levels=[0.5], colors='w', linewidths=0.4)
                ax.contour(cen, cen, (lab == 2).T.astype(float), levels=[0.5], colors='#40c0ff', linewidths=0.6)
                ax.set_title(f'{N}$^{D}$ (l/h = {lh[N]:g}), $\\Delta V/V$ = {lev}', fontsize=12)
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]' if D == 2 else 'y [mm] (z = 50 mm)')
        fig.colorbar(im, ax=axes, shrink=0.5, label='damage d')
        save(fig, f'asrmesh_{D}D_damage.png')
