"""
Slide figure: mean CG per elastic solve vs grid, Green / Green-Jacobi / dual (x2), fixed interface width (l = 4, 10 nm)
and constant l/h (l = 2h, 4h). y axis capped at 600 (Green-Jacobi at 512^2, l = 2h and 4h, is above it).
The 1 nm and 0.5 nm points use dt = 1e-7 s (see README: the evolution is slowed at sharp l).
    python plot_sintering_slide_const_lh.py  ->  figures/slide_sintering_mesh_const_lh.{png,pdf} + talk figures/
"""
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
GRIDS = [32, 64, 128, 256, 512, 1024]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})


def cg(P, N, W):
    suf = '' if W == 4 else f'_W{W:g}'
    f = os.path.join(HERE, 'exp_data', f'SINT_N{N}_{P}_W{W:g}_P50u_k0.0001{suf}_steps.npz')
    if not os.path.exists(f):
        return None
    s = np.load(f, allow_pickle=True)
    if s['t'][-1] < 4.9e-4:                                       # unfinished run
        return None
    return float(np.nanmean(s['cg_mean'])) * (2.0 if P == 'dual' else 1.0)


FAMILIES = [('Green', 'Green', '#7fc97f', '#0b6b0b'), ('Green_Jacobi', 'Green-Jacobi', '#7fb2e6', '#0b3d91'),
            ('dual', r'dual ($\times 2$)', '#f4a0a0', '#a50f15')]
SERIES = [(r'fixed $l = 4$ nm', lambda N: 4.0, '-', 'o', False, False), (r'fixed $l = 10$ nm', lambda N: 10.0, '--', '^', False, True),
          (r'$l = 2h$', lambda N: 2 * 128 / N, '-', 's', True, False), (r'$l = 4h$', lambda N: 4 * 128 / N, '--', 'D', True, True)]
fig, ax = plt.subplots(figsize=(4.8, 4.3), layout='constrained')
for P, name, light, dark in FAMILIES:
    for lab, width, ls, mk, strong, hollow in SERIES:
        pts = [(N, cg(P, N, width(N))) for N in GRIDS]
        pts = [(a, b) for a, b in pts if b is not None]
        col = dark if strong else light
        ax.plot(*zip(*pts), ls + mk, color=col, mfc='white' if hollow else col, mew=1.3, ms=6 if strong else 4.5,
                lw=2.2 if strong else 1.4, zorder=5 if strong else 3)
        print(name, lab, [(a, round(b)) for a, b in pts])
ax.set_xscale('log', base=2)
ax.set_xticks(GRIDS), ax.set_xticklabels([f'${n}^2$' for n in GRIDS]), ax.minorticks_off()
ax.set_xlim(32, 1024), ax.set_ylim(0, 600)
ax.set(xlabel='grid size', ylabel='CG iterations per solve')
ax.grid(alpha=0.3)
hand = [Line2D([], [], color=d, lw=2.5, label=n) for _, n, _, d in FAMILIES]
hand += [Line2D([], [], color='0.25' if s else '0.6', ls=ls, marker=m, mfc='white' if h else ('0.25' if s else '0.6'),
                lw=2.2 if s else 1.4, ms=6, label=lab) for lab, _, ls, m, s, h in SERIES]
fig.legend(handles=hand, loc='outside upper center', ncol=3, fontsize=10, frameon=False)
out = os.path.join(HERE, 'figures', 'slide_sintering_mesh_const_lh')
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=200)
    shutil.copy(f'{out}.{ext}', TALK)
print('wrote', out)
