"""Periodic open channel (3D, h = 1/32): follower vs dead pressure and follower with p(z) = p (1 + 0.5 cos 2 pi z/Lz).
Pressure vs cavity volume and Hessian products per Newton solve.  Usage: python plot_channel3d_compare.py"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
CASES = [('per', 'follower pressure, uniform', dict(color='#00441b', ls='-', marker='o', mfc='#00441b')),
         ('per_dead', 'dead load (nodal forces), uniform', dict(color='#41ab5d', ls='--', marker='s', mfc='white')),
         ('per_pv0.5', 'follower, p(z) = p (1 + 0.5 cos 2 pi z / Lz)', dict(color='#238b45', ls='-.', marker='^',
                                                                            mfc='#238b45'))]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for tag, label, st in CASES:
    run = sorted(glob.glob(os.path.join(HERE, 'runs', f'bar3d_n32_R0.3125_p12_{tag}_G_wall_gpu_*')),
                 key=os.path.getmtime)[-1]
    d = np.load(os.path.join(run, 'state.npz'))
    axes[0].plot(d['volume'] / d['volume'][0], d['p'], ms=6, lw=1.8, label=label, **st)
    axes[1].plot(d['p'][1:], d['hessp'][1:] / d['nit'][1:], ms=6, lw=1.8, label=label, **st)
    print(f'{label:48s} volume x{d["volume"][-1] / d["volume"][0]:.3f}  min J {d["min_J"][-1]:.3f}  '
          f'{d["hessp"].sum() / d["nit"].sum():.1f} hessp/solve, {d["nit"].sum()} solves')
axes[0].set_xlabel('cavity volume / initial volume')
axes[0].set_ylabel('mean pressure p   (E = 100)')
axes[0].set_xlim(0, None)
axes[0].set_ylim(0, None)
axes[0].grid(alpha=0.3)
axes[1].set_xlabel('mean pressure p')
axes[1].set_ylabel('Hessian products per Newton solve')
axes[1].set_xlim(0, 12.5)
axes[1].set_ylim(0, None)
axes[1].grid(alpha=0.3)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.02))
fig.suptitle('periodic open channel (3D), x walls clamped, h = 1/32, corrected Green (wall FFT)')
fig.tight_layout()
out = os.path.join(HERE, 'channel3d_compare.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
print(out)
