"""
Slide figure of the sintering mesh study: mean CG per elastic solve over the run against the grid, Green / Green-Jacobi /
dual, interface width W = 10 nm (Seiz et al. 2024; solid) and W = 4 nm (dashed). LaTeX fonts, slide size.
    python plot_sintering_slide_mesh.py  ->  figures/slide_sintering_mesh.{png,pdf} (+ copy to the talk figures/)
    BOTH_L=1 (env): talk notation, l = 4 nm solid + l = 10 nm dashed -> *_both_l.{png,pdf}
    NO_DUAL=1 (env): without the dual lines -> *_nodual.{png,pdf}
    ONLY_W4=1 (env): only W = 4 nm, solid lines -> *_W4.{png,pdf}
    DUAL_FACTOR=2 (env): dual iterations counted x2 (cost of a dual iteration) -> slide_sintering_mesh_dualx2.{png,pdf}
"""
import glob
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
PRE = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
       ('dual', 'dual', 'tab:red', '^')]
NO_DUAL = os.environ.get('NO_DUAL') == '1'
if NO_DUAL:
    PRE = PRE[:2]
GRIDS = [32, 64, 128, 256, 512, 1024]
DUAL_FACTOR = float(os.environ.get('DUAL_FACTOR', 1))
ONLY_W4 = os.environ.get('ONLY_W4') == '1'
BOTH_L = os.environ.get('BOTH_L') == '1'
CASES = [('W4_P50u_k0.0001', '-', None, 4), ('W10_P50u_k0.0001_W10', '--', 'white', 10)] if BOTH_L else [('W4_P50u_k0.0001', '-', None, 4)] if ONLY_W4 else [('W10_P50u_k0.0001_W10', '-', None, 10), ('W4_P50u_k0.0001', '--', 'white', 4)]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, ax = plt.subplots(figsize=(4.4, 3.6), layout='constrained')
for P, lab, col, mk in PRE:
    for pat, ls, mfc, w in CASES:
        mfc = mfc or col
        pts = []
        for N in GRIDS:
            f = os.path.join(HERE, 'exp_data', f'SINT_N{N}_{P}_{pat}_steps.npz')
            if os.path.exists(f):
                st = np.load(f, allow_pickle=True)
                if st['t'][-1] < 4.9e-4:                 # run not finished yet (early steps are the expensive ones)
                    continue
                pts.append((N, np.nanmean(st['cg_mean']) * (DUAL_FACTOR if P == 'dual' else 1.0)))
        ax.plot(*zip(*pts), ls + mk, color=col, lw=2.2, ms=6, mfc=mfc, mew=1.6, clip_on=False, zorder=5,
                label=lab if (w == 4 if BOTH_L else (w == 10 or ONLY_W4)) else None)
        print(P, w, [round(v) for _, v in pts])
if BOTH_L:
    ax.plot([], [], 'k-', lw=2.2, label='$l = 4$ nm')
    ax.plot([], [], 'k--', lw=2.2, label='$l = 10$ nm')
elif not ONLY_W4:
    ax.plot([], [], 'k-', lw=2.2, label='$W = 10$ nm')
    ax.plot([], [], 'k--', lw=2.2, label='$W = 4$ nm')
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024), ax.set_ylim(0, 400)
ax.set(xlabel='grid size', ylabel=r'CG per solve (dual $\times 2$)' if DUAL_FACTOR == 2 else 'CG iterations per solve')
ax.grid(alpha=0.3)
if BOTH_L:                                                       # first row preconditioners, second row l
    h, t = ax.get_legend_handles_labels()
    blank = plt.Line2D([], [], alpha=0)
    o = [0, 2, 1, 3] if NO_DUAL else [0, 3, 1, 4, 2]
    fig.legend([h[i] for i in o] + ([] if NO_DUAL else [blank]), [t[i] for i in o] + ([] if NO_DUAL else ['']),
               loc='outside upper center', ncol=2 if NO_DUAL else 3, fontsize=11, frameon=False)
else:
    fig.legend(loc='outside upper center', ncol=3, fontsize=10, frameon=False)
out = os.path.join(HERE, 'figures', 'slide_sintering_mesh' + ('_dualx2' if DUAL_FACTOR == 2 else '') + ('_W4' if ONLY_W4 else '') + ('_both_l' if BOTH_L else '') + ('_nodual' if NO_DUAL else ''))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)
