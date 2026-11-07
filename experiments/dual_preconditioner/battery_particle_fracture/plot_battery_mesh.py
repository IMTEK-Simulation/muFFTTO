"""Mesh-size study of the NEMO battery runs (sbatch_battery_gpu.sh; CSVs synced to exp_data/nemo/):
mean CG iterations per elastic and per diffusion solve vs l/h (fixed l = 0.23 um in the core, h = 6 um / N), before
and during crack growth, Green / Green-Jacobi / dual, 2D and 3D. Grids with l/h < 2 are under-resolved (open markers,
grey band). Windows "before" / "during" from the dual run of the same grid (identical trajectories).

    python plot_battery_mesh.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
data = os.path.join(here, 'exp_data', 'nemo')
figs = os.path.join(here, 'figures')
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 14, 'axes.titlesize': 14, 'legend.fontsize': 12})
PRECS = ['Green', 'Green_Jacobi', 'dual']
COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
L_CORE, L_CELL = 0.23, 6.0
GRIDS = {2: [32, 64, 128, 256, 512, 1024], 3: [32, 64, 128, 256]}


def load(dim, N, P):
    base = os.path.join(data, f'battery_N{N}_{P}_ms0.01{"_3D" if dim == 3 else ""}_tu')
    if not os.path.exists(base + '_steps.csv'):
        return None
    steps = np.atleast_1d(np.genfromtxt(base + '_steps.csv', delimiter=',', names=True, dtype=None, encoding=None))
    its = np.atleast_1d(np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None,
                                      missing_values='', filling_values=-1))
    return None if steps.size < 2 else dict(steps=steps, its=its)


def windows(steps):
    a = steps['a_c']
    g = a - a[0]
    on = g > 0.002
    idx = np.arange(len(a))
    if not on.any():
        return idx >= 0, idx < 0
    i0 = np.argmax(on)
    i1 = np.argmax(g >= 0.95 * g[-1])
    return idx < i0, (idx >= i0) & (idx <= max(i1, i0))


def finished(dim, N, P):
    for name in (f'battery_tu_N{N}_{P}{"_3D" if dim == 3 else ""}{s}.log' for s in ('', '_r1', '_r2')):
        f = os.path.join(here, 'logs', 'nemo', name)
        if os.path.exists(f) and 'Elapsed time' in open(f).read():
            return True
    return False


rows = []
fig, axes = plt.subplots(2, 4, figsize=(22, 9.5), constrained_layout=True)
for r, dim in enumerate((2, 3)):
    for w, wname in enumerate(('before', 'during')):
        for kind, col in (('elastic', w), ('diffusion', 2 + w)):
            ax = axes[r, col]
            ax.axvspan(0, 2, color='0.9', zorder=0)
            for P in PRECS:
                xs, ys, done = [], [], []
                for N in GRIDS[dim]:
                    ref = load(dim, N, 'dual')
                    run = load(dim, N, P) if kind == 'elastic' else ref
                    if ref is None or run is None:
                        continue
                    mask = windows(ref['steps'])[w]
                    sel = np.isin(run['its']['step'], ref['steps']['step'][mask])
                    col_name = 'cg_u' if kind == 'elastic' else f'cg_c_{P}'
                    vals = run['its'][col_name][sel]
                    vals = vals[vals >= 0]
                    if len(vals) == 0:
                        continue
                    xs.append(L_CORE / (L_CELL / N))
                    ys.append(float(np.mean(vals)))
                    done.append(finished(dim, N, P if kind == 'elastic' else 'dual'))
                    rows.append((dim, N, kind, wname, P, ys[-1], len(vals), done[-1]))
                if not xs:
                    continue
                ax.plot(xs, ys, color=COLOR[P], lw=1.5, label=LABEL[P])
                for x, y, d in zip(xs, ys, done):
                    ax.plot(x, y, 'o', color=COLOR[P], mfc=COLOR[P] if x >= 2 else 'white', ms=8,
                            alpha=1.0 if d else 0.45)
            ax.set_xscale('log', base=2)
            ax.set_xticks([1.23, 2.45, 4.9, 9.8, 19.6, 39.2])
            ax.set_xticklabels(['1.2', '2.5', '4.9', '9.8', '20', '39'])
            ax.set_xlim(1, 50)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('l / h  (core l = 0.23 um)')
            ax.set_ylabel(f'mean CG per {kind} solve')
            ax.set_title(f'{dim}D, {kind}, {wname} crack growth')
            ax.grid(alpha=0.3)
            ax.legend(loc='best')
fig.suptitle('core-shell particle, Tu et al. parameters, fixed l: mesh refinement (grey: l/h < 2 under-resolved; '
             'open marker: under-resolved, faded: run not finished)')
out = os.path.join(figs, 'battery_mesh_study_nemo.png')
fig.savefig(out, dpi=130)
print('saved', out)
for row in rows:
    print(f'{row[0]}D N={row[1]:5d} {row[2]:9s} {row[3]:7s} {LABEL[row[4]]:13s} mean CG = {row[5]:7.1f} '
          f'({row[6]} solves){"" if row[7] else "  [running]"}')
if os.environ.get('NO_OPEN') != '1':
    os.system(f'xdg-open {out} >/dev/null 2>&1 &')
