"""
Mesh study of the cantilever fracture runs (cantilever_fracture.py): corrected Green against corrected dual for the
elastic solves, n_H = 32, 64, 128 (l fixed, so the same physical problem on every mesh). Unfinished runs are plotted
as far as they got.

    top     tip force vs tip deflection;  CG iterations per elastic solve along the run (vs tip deflection);
            CG iterations per damage solve along the run
    bottom  elastic CG iterations in total;  mean elastic CG iterations per solve;  wall time of the elastic solves

usage:  [MODEL=amor|hybrid] python cantilever_fracture_study.py      -> exp_data/cantilever_fracture_study[_amor].png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.realpath(__file__))
OUT = os.path.join(HERE, 'exp_data', 'fracture')
model = os.environ.get('MODEL', 'amor')
TAIL = '_holes4_het10_l0.0125_ft0.01' + ('_amor' if model == 'amor' else '')
MESHES = [32, 64, 128]
LABELS = {'green_c': 'Green, corrected', 'dual_c': 'dual, corrected'}
# preconditioner = hue (Green blue, dual orange), mesh = shade (light coarse -> dark fine) and marker
SHADES = {'green_c': ['#86b6ef', '#3d7fd0', '#184f95'], 'dual_c': ['#f4a582', '#e0603a', '#a8340f']}
MARKERS = ['o', 's', '^']
PLAIN = FuncFormatter(lambda v, _: f'{v:.6g}')


def load(n, p):
    import netCDF4
    f = os.path.join(OUT, f'nH{n}_{p}{TAIL}.nc')
    if not os.path.exists(f):
        return None
    try:
        with netCDF4.Dataset(f) as ds:
            r = {k: np.asarray(v[:]) for k, v in ds.variables.items() if v.dimensions == ('frame',)}
            r.update({k: ds.getncattr(k) for k in ('target_crit', 'force_crit', 'h')})
    except OSError:  # being written
        return None
    return r if len(r['step']) else None


fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout='constrained')
totals = {p: [] for p in LABELS}
for p in LABELS:
    for i, n in enumerate(MESHES):
        r = load(n, p)
        if r is None:
            continue
        kw = dict(color=SHADES[p][i], label=f'{LABELS[p]}, $n_H$ = {n}')
        conv = r['converged'] > 0
        tc, fc = float(r['target_crit']), float(r['force_crit'])
        x = r['target']  # absolute tip deflection: the first-f_t scale depends on the mesh (pixelated stress peaks)
        axes[0, 0].plot(x[conv], 1e5 * r['force'][conv], lw=1.8, ls='-' if p == 'dual_c' else (0, (4, 2)), **kw)
        axes[0, 1].plot(x, r['cg_u'], lw=1, marker=MARKERS[i], ms=2.5, mew=0, **kw)
        axes[0, 2].plot(x, r['cg_d'], lw=1, marker=MARKERS[i], ms=2.5, mew=0, **kw)
        done = (r['force'][-1] < 0.05 * r['force'].max() and r['d_max'][-1] > 0.9) or x[-1] / tc >= 5.9  # 6 x: the cap
        totals[p].append((n, r['cg_u'].sum(), r['cg_u'].mean(), r['time_u'].sum(), x[-1], done))
for p, rows in totals.items():
    if not rows:
        continue
    a = np.array(rows, dtype=float)
    kw = dict(color=SHADES[p][2], marker='o' if p == 'green_c' else '^', ms=8, lw=2, label=LABELS[p],
              ls='-' if p == 'dual_c' else (0, (4, 2)))
    for ax, col in zip(axes[1], (1, 2, 3)):
        ax.plot(a[:, 0], a[:, col], **kw)
        for n, v, fin in zip(a[:, 0], a[:, col], a[:, 5]):
            if not fin:  # unfinished run: open marker
                ax.plot(n, v, kw['marker'], ms=11, mfc='white', mec=kw['color'], mew=2)
axes[0, 0].set(xlabel='tip deflection (beam length 1)', ylabel='tip force × $10^5$ (E = 1, per unit thickness)',
               title='load-deflection (dual solid, Green dashed)')
axes[0, 1].set(xlabel='tip deflection', ylabel='CG iterations',
               title='elastic solve: CG iterations per solve, every staggered iteration')
axes[0, 2].set(xlabel='tip deflection', ylabel='CG iterations',
               title='damage solve (Green-Jacobi): CG iterations per solve')
axes[1, 0].set(ylabel='CG iterations', title='elastic CG iterations, whole run')
axes[1, 1].set(ylabel='CG iterations', title='mean elastic CG iterations per solve')
axes[1, 2].set(ylabel='wall time [s]', title='wall time of the elastic solves (32, 64: CPU, 128: GPU, shared)')
for ax in axes.flat:
    ax.grid(True, color='0.9')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(PLAIN)
for ax in axes[1]:
    ax.set_xscale('log', base=2)
    ax.set_xticks(MESHES, [str(m) for m in MESHES])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('$n_H$ (pixels across the beam height)')
axes[0, 0].legend(frameon=False, fontsize=9)
axes[1, 0].legend(frameon=False, fontsize=10, title='open marker: run not finished', title_fontsize=9)
fig.suptitle('AT2 fracture of the clamped heterogeneous cantilever with 4 holes (contrast 10, $\\ell$ = 0.0125 fixed), '
             + ('Amor split in the stiffness (closed cracks carry compression)' if model == 'amor'
                else 'hybrid (isotropic degradation)') + ': mesh study of the elastic preconditioner', fontsize=13)
out = os.path.join(HERE, 'exp_data', 'cantilever_fracture_study' + ('_amor' if model == 'amor' else '') + '.png')
fig.savefig(out, dpi=120)
print(f'wrote {out}')
