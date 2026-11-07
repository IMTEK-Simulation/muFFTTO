"""Figures of the PF-CZM concrete mesh-size study (NEMO runs, copied to exp_data_nemo/ and logs_nemo/).

python plot_pfczm_mesh.py [LOAD_DIR=xx]
  figures/PFCZM_mesh_cg_vs_bh_<dir>.png       mean CG per elastic solve vs b/h, before the peak / during cracking, 2D + 3D
  figures/PFCZM_mesh_stress_strain_<dir>.png  stress-strain per grid (all preconditioners), 2D + 3D
  figures/PFCZM_mesh_cg_vs_load_2D_<dir>.png  mean CG per solve vs load, one panel per 2D grid
  figures/PFCZM_mesh_cracks_2D_<dir>.png      final phase field of the dual runs per 2D grid (if frames were copied)
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data_nemo')
load_dir = sys.argv[1] if len(sys.argv) > 1 else 'xx'
B_MM, CELL_MM, E_UNIT = 1.5, 100.0, 20e3
precs = ['Green', 'Green_Jacobi', 'dual']
color = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
label = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
grids = {2: [32, 64, 128, 256, 512, 1024], 3: [32, 64, 128, 256]}
lstyle = {32: ':', 64: '-.', 128: '--', 256: (0, (5, 1, 1, 1, 1, 1)), 512: '-', 1024: (0, (8, 2))}
comp = r'\bar\varepsilon_{xx}' if load_dir == 'xx' else r'\bar\gamma_{xy}'
scomp = r'\bar\sigma_{xx}' if load_dir == 'xx' else r'\bar\sigma_{xy}'
plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13})
fig_dir = os.path.join(here, 'figures')


def bh(N):
    return B_MM / (CELL_MM / N)


def load(D, N, P):
    # NEMO runs (exp_data_nemo) or laptop CPU runs of the same script (exp_data, 2D 32-128²)
    for folder in (data, os.path.join(here, 'exp_data')):
        stem = os.path.join(folder, f'PFCZM{D}D_N{N}_{P}_{load_dir}_b1.5mm_k1e-05')
        if os.path.exists(stem + '_steps.csv'):
            break
    else:
        return None
    st = np.genfromtxt(stem + '_steps.csv', delimiter=',', names=True)
    it = np.genfromtxt(stem + '_iterations.csv', delimiter=',', names=True)
    if st.size < 3:
        return None
    # restarts: keep the last occurrence of every (step) / (step, staggered iteration)
    _, keep = np.unique(st['step'][::-1], return_index=True)
    st = st[::-1][keep]
    key = it['step'] * 10000 + it['staggered_iteration']
    _, keep = np.unique(key[::-1], return_index=True)
    it = it[::-1][keep]
    return st, it, stem


def save(fig, name):
    path = os.path.join(fig_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('saved', path)


def phase_means(st, it):
    peak = st['load'][int(np.argmax(st['stress']))]
    pre, post = it['cg_u'][it['load'] <= peak + 1e-12], it['cg_u'][it['load'] > peak + 1e-12]
    return pre.mean(), (post.mean() if post.size else np.nan)


runs = {(D, N, P): load(D, N, P) for D in (2, 3) for N in grids[D] for P in precs}

# 1. mean CG per elastic solve vs b/h
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
print('DIM N b/h P  mean CG before peak, during cracking, solves, last load step, finished')
for ax, D in zip(axes, (2, 3)):
    for P in precs:
        xs, pre, post = [], [], []
        for N in grids[D]:
            r = runs[(D, N, P)]
            if r is None:
                continue
            done = r[0]['step'][-1] >= 299 or r[0]['stress'][-1] < 0.02 * r[0]['stress'].max()
            if not done:                     # running: shown only in the per-load plot
                print(f'{D} {N:5d} {P:13s} still running (load step {int(r[0]["step"][-1])})')
                continue
            a, c = phase_means(r[0], r[1])
            xs.append(bh(N)); pre.append(a); post.append(c)
            print(f'{D} {N:5d} {bh(N):5.2f} {P:13s} {a:7.1f} {c:7.1f} {len(r[1]):6d} {int(r[0]["step"][-1]):4d}')
        ax.plot(xs, pre, 'o--', color=color[P], mfc='white', ms=8, label=f'{label[P]}, before peak')
        ax.plot(xs, post, 's-', color=color[P], ms=8, label=f'{label[P]}, cracking')
    ax.axvspan(0.3, 2.0, color='0.9', zorder=0)
    ax.text(0.02, 0.02, 'grey: b/h < 2, under-resolved', transform=ax.transAxes, va='bottom', fontsize=11, color='0.35')
    ax.set_xscale('log', base=2)
    ticks = [bh(N) for N in grids[D]]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.2g}'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(0.38, ticks[-1] * 1.25)
    ax.set_ylim(0)
    ax.set_xlabel(f'b/h (b = 1.5 mm fixed, grids {grids[D][0]}..{grids[D][-1]}' + ('²' if D == 2 else '³') + ')')
    ax.set_ylabel('mean CG iterations per elastic solve')
    ax.set_title(f'{D}D PF-CZM concrete, {"tension" if load_dir == "xx" else "shear"}')
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=10, loc='upper left', bbox_to_anchor=(0.0, 0.9))
fig.tight_layout()
save(fig, f'PFCZM_mesh_cg_vs_bh_{load_dir}.png')

# 2. stress-strain per grid
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, D in zip(axes, (2, 3)):
    for N in grids[D]:
        for P in precs:
            r = runs[(D, N, P)]
            if r is None:
                continue
            st = r[0]
            ax.plot(np.r_[0, st['load']] * 1e3, np.r_[0, st['stress']] * E_UNIT, ls=lstyle[N], color=color[P], lw=1.5,
                    label=f'{N}{"²" if D == 2 else "³"} (b/h {bh(N):.2g})' if P == 'dual' else None)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(rf'${comp}$ ($\times 10^{{-3}}$)')
    ax.set_ylabel(rf'${scomp}$ (MPa)')
    ax.set_title(f'{D}D: line style = grid, colour = preconditioner')
    ax.grid(alpha=0.3)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=10)
fig.tight_layout()
save(fig, f'PFCZM_mesh_stress_strain_{load_dir}.png')

# 3. CG per solve vs load, 2D grids
avail = [N for N in grids[2] if any(runs[(2, N, P)] is not None for P in precs)]
if avail:
    nc = min(3, len(avail))
    nr = (len(avail) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 4.3 * nr), squeeze=False)
    for ax, N in zip(axes.flat, avail):
        for P in precs:
            r = runs[(2, N, P)]
            if r is not None:
                ax.plot(r[0]['load'] * 1e3, r[0]['mean_cg_u'], '-', color=color[P], label=label[P], lw=1.4)
        ax.set_title(f'2D {N}², b/h = {bh(N):.2g}' + (' (under-resolved)' if bh(N) < 2 else ''))
        ax.set_xlabel(rf'${comp}$ ($\times 10^{{-3}}$)')
        ax.set_ylabel('mean CG per elastic solve')
        ax.set_xlim(0)
        ax.set_ylim(0)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(avail):]:
        ax.axis('off')
    axes[0][0].legend()
    fig.tight_layout()
    save(fig, f'PFCZM_mesh_cg_vs_load_2D_{load_dir}.png')

# 4. final crack pattern (dual) per 2D grid
frames = []
for N in avail:
    r = runs[(2, N, 'dual')]
    if r is None:
        continue
    files = sorted(glob.glob(r[2] + '_frames/d_*.npy'))
    if files:
        frames.append((N, np.load(files[-1]).astype(float), np.load(r[2] + '_phase.npy'), files[-1]))
if frames:
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames), 4.4), squeeze=False, layout='constrained')
    for ax, (N, d, ph, fn) in zip(axes[0], frames):
        im = ax.imshow(d.T, origin='lower', extent=(0, CELL_MM, 0, CELL_MM), vmin=0, vmax=1, cmap='inferno',
                       interpolation='nearest')
        xc = (np.arange(N) + 0.5) * CELL_MM / N
        ax.contour(xc, xc, ph.T, [0.5], colors='cyan', linewidths=0.5)
        ax.set_title(f'{N}², b/h = {bh(N):.2g}, step {os.path.basename(fn)[2:6]}', fontsize=12)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
    fig.colorbar(im, ax=axes[0], shrink=0.8, label='phase field d')
    save(fig, f'PFCZM_mesh_cracks_2D_{load_dir}.png')
