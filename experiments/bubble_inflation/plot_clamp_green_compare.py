"""Clamped pressurised-hole plate (plate 0.875, void t = 1/16), Green preconditioner: dense corrected (real clamp dofs
only), wall-FFT corrected, projected. One panel per mesh (Hessian products per increment vs p) + totals vs mesh.
Usage: python plot_clamp_green_compare.py   (latest runs/hole_n<N>_R0.25_p30_t0.0625_clampLR_<P>_* per mesh)"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
meshes = (32, 64, 128)
VARIANTS = {   # Green family = green; variants: shade + line style + marker + fill
    'G_corr': dict(color='#00441b', ls='-', marker='o', mfc='#00441b', label='corrected Green, dense S (clamp dofs only)'),
    'G_wall': dict(color='#238b45', ls='-.', marker='^', mfc='#238b45', label='corrected Green, wall FFT'),
    'G': dict(color='#74c476', ls='--', marker='s', mfc='white', label='projected Green'),
}


def load(n, P, t='0.0625'):
    runs = sorted(glob.glob(os.path.join(HERE, 'runs', f'hole_n{n}_R0.25_p30_t{t}_clampLR_{P}_2*')))
    return np.load(os.path.join(runs[-1], 'state.npz'))


fig, axes2 = plt.subplots(2, 4, figsize=(20, 10.4))
axes, axes_s = axes2[0], axes2[1]
totals = {P: [] for P in VARIANTS}
per_solve = {P: [] for P in VARIANTS}
for ax, n in zip(axes[:3], meshes):
    for P, st in VARIANTS.items():
        d = load(n, P)
        ax.plot(d['p'][1:], d['hessp'][1:], ms=5, lw=1.6, **st)
        totals[P].append(int(d['hessp'].sum()))
        axes_s[list(meshes).index(n)].plot(d['p'][1:], d['hessp'][1:] / d['nit'][1:], ms=5, lw=1.6, **st)
        per_solve[P].append(d['hessp'].sum() / d['nit'].sum())
    ax.set_title(f'h = 1/{n}  (cell {n}$^2$ px)')
    ax.set_xlabel('pressure p   (E = 100)')
    ax.set_ylabel('Hessian products per increment')
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 400)
    ax.grid(alpha=0.3)
ax = axes[3]
for P, st in VARIANTS.items():
    ax.plot(meshes, totals[P], ms=8, lw=2, **st)
    for n, v in zip(meshes, totals[P]):
        ax.annotate(f'{v}', (n, v), textcoords='offset points', xytext={'G_corr': (-14, 10), 'G_wall': (-14, -18), 'G': (-14, 10)}[P], fontsize=11, color=st['color'])
ax.set_xticks(meshes, [str(n) for n in meshes])
ax.set_xlabel('pixels per unit length (1/h)')
ax.set_ylabel('Hessian products, whole load path')
ax.set_title('total, p = 0 -> 30')
ax.set_xlim(0, 140)
ax.set_ylim(0, None)
ax.grid(alpha=0.3)
for ax, n in zip(axes_s[:3], meshes):
    ax.set_title(f'per Newton solve, h = 1/{n}')
    ax.set_xlabel('pressure p   (E = 100)')
    ax.set_ylabel('Hessian products per solve')
    ax.set_xlim(0, 31)
    ax.set_ylim(0, 120)
    ax.grid(alpha=0.3)
ax = axes_s[3]
for P, st in VARIANTS.items():
    ax.plot(meshes, per_solve[P], ms=8, lw=2, **st)
    for n, v in zip(meshes, per_solve[P]):
        ax.annotate(f'{v:.1f}', (n, v), textcoords='offset points',
                    xytext={'G_corr': (-14, 10), 'G_wall': (-14, -18), 'G': (-14, 10)}[P], fontsize=11, color=st['color'])
ax.set_xticks(meshes, [str(n) for n in meshes])
ax.set_xlabel('pixels per unit length (1/h)')
ax.set_ylabel('Hessian products per solve (mean)')
ax.set_title('mean per Newton solve')
ax.set_xlim(0, 140)
ax.set_ylim(0, 120)
ax.grid(alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.01), fontsize=13)
fig.suptitle('pressurised hole, left + right faces clamped, Green preconditioner (C_ref = solid), '
             'plate 0.875, void t = 1/16 (physical)')
fig.tight_layout()
out = os.path.join(HERE, 'clamp_green_compare.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(out)
for P in VARIANTS:
    print(P, totals[P])
