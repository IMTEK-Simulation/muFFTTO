"""Void-thickness sweep (run_void_sweep.sh): wall-FFT corrected Green, clamped pressurised-hole plate, plate 0.875
fixed, void t = 1/32 ... 1/4 (cell grows). Totals vs mesh per t, per-increment counts at h = 1/128, and the check
that the solution does not depend on t (hole area at p = 30).  Usage: python plot_void_sweep.py"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
meshes = (32, 64, 128)
T = {'0.03125': ('1/32', '#a1d99b', ':', 'v', 'white'),
     '0.0625': ('1/16', '#41ab5d', '--', 's', 'white'),
     '0.125': ('1/8', '#238b45', '-.', '^', '#238b45'),
     '0.25': ('1/4', '#00441b', '-', 'o', '#00441b')}


def load(n, t):
    runs = sorted(glob.glob(os.path.join(HERE, 'runs', f'hole_n{n}_R0.25_p30_t{t}_clampLR_G_wall_2*')))
    return np.load(os.path.join(runs[-1], 'state.npz'))


fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
print('t      1/h   total hessp   hole area at p=30   min J solid   cell px')
for t, (lab, c, ls, mk, mfc) in T.items():
    tot = []
    for n in meshes:
        d = load(n, t)
        tot.append(int(d['hessp'].sum()))
        print(f'{lab:5s} {n:5d} {tot[-1]:10d} {d["area"][-1]:16.5f} {d["min_J"][-1]:13.3f} {int(d["ncell"]):8d}')
        if n == 128:
            axes[1].plot(d['p'][1:], d['hessp'][1:], color=c, ls=ls, marker=mk, mfc=mfc, ms=5, lw=1.6,
                         label=f't = {lab}')
    axes[0].plot(meshes, tot, color=c, ls=ls, marker=mk, mfc=mfc, ms=8, lw=2, label=f't = {lab}')
axes[0].set_xticks(meshes, [str(n) for n in meshes])
axes[0].set_xlabel('pixels per unit length (1/h)')
axes[0].set_ylabel('Hessian products, whole load path')
axes[0].set_xlim(0, 140)
axes[0].set_ylim(0, None)
axes[0].grid(alpha=0.3)
axes[0].set_title('total, p = 0 -> 30')
axes[1].set_xlabel('pressure p   (E = 100)')
axes[1].set_ylabel('Hessian products per increment')
axes[1].set_xlim(0, 31)
axes[1].set_ylim(0, None)
axes[1].grid(alpha=0.3)
axes[1].set_title('h = 1/128')
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
fig.suptitle('void thickness t (physical), wall-FFT corrected Green, clamped plate 0.875 with pressurised hole')
fig.tight_layout()
out = os.path.join(HERE, 'void_sweep.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(out)
