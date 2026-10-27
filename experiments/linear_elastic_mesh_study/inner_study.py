"""
Corrected Green with S solved by an inner CG instead of assembling it (runs by run_inner.sh). Beam with 4 holes,
void 0.00625, clamped face.

    1. outer CG iterations vs h: inner CG for each inner tolerance, assembled S, plain Green
    2. inner CG iterations per outer iteration (mean and max) vs h
    3. Green applications in total (setup + CG) vs h
    4. wall time (setup + CG) vs h

usage:  python inner_study.py
writes exp_data/inner_study.png
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
RUNS = os.path.join(HERE, 'exp_data', 'runs')
MESHES = (32, 64, 128, 256, 512, 1024)
TOLS = ('1e-08', '1e-04', '0.01')
SUFFIX = '_void0.00625{}_holes4r0.05.npz'


def load(n_H, tag):
    f = os.path.join(RUNS, f'n_H{n_H}_cgtol1e-06' + SUFFIX.format(tag))
    return np.load(f) if os.path.exists(f) else None


def series(tag):
    runs = [(n, load(n, tag)) for n in MESHES]
    return [(n, r) for n, r in runs if r is not None]


cases = [(f'inner CG, tol {float(t):g}', f'_pgdi{float(t):g}', f'C{k}', '-', 'o') for k, t in enumerate(TOLS)]
cases += [('assembled S', '_pgd', 'k', '-', 's'), ('plain Green', '', '0.5', '--', 'v')]

fig, axes = plt.subplots(1, 4, figsize=(20, 5.2), layout='constrained')
for label, tag, c, ls, mk in cases:
    runs = series(tag)
    if not runs:
        continue
    h = np.array([float(r['h']) for _, r in runs])
    its = np.array([int(r['cg_its']) for _, r in runs])
    axes[0].loglog(h, its, ls, color=c, marker=mk, label=label)
    m = np.array([int(r['nb_clamped_nodes']) * 2 for _, r in runs])
    if tag.startswith('_pgdi'):
        inner = [r['inner_its'] for _, r in runs]
        axes[1].loglog(h, [i.mean() for i in inner], ls, color=c, marker=mk, label=f'{label}, mean')
        axes[1].loglog(h, [i.max() for i in inner], ':', color=c, marker=mk, mfc='none')
        green = np.array([i.sum() + 2 * len(i) for i in inner])  # inner steps + the 2 Green applications per call
    elif tag == '_pgd':
        green = m + 2 * its  # assembly: one per fixed dof
    else:
        green = its
    axes[2].loglog(h, green, ls, color=c, marker=mk, label=label)
    axes[3].loglog(h, [float(r['time']) + float(r['setup_time']) for _, r in runs], ls, color=c, marker=mk, label=label)

axes[1].plot([], [], ':', color='0.3', marker='o', mfc='none', label='max (dotted)')
titles = ('outer CG iterations', 'inner CG iterations per outer iteration', 'Green applications, setup + CG',
          'wall time, setup + CG [s]')
for ax, t in zip(axes, titles):
    ax.set_title(t)
    ax.set_xlabel('pixel size h')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
fig.suptitle('beam with holes, void 0.00625, clamped face: corrected Green with S solved by an inner CG '
             '(4-6 runs in parallel, times rough)', fontsize=12)
out = os.path.join(HERE, 'exp_data', 'inner_study.png')
fig.savefig(out, dpi=150)
print(f'wrote {out}')
