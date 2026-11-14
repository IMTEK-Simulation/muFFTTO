"""Objective and volume fraction over the L-BFGS iterations of the 3D pocket tests (64^3, sine start, Green-Jacobi,
eta = 1/16) against the reference a = 5, w = 0.2 (start_sine, 3 segments). History frames every 10 iterations.
    python plot_TO3D_pockets_history.py  ->  figures/TO3D_pockets_history.png
"""
import glob
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'example_3D_elasticity_TO')
P = os.path.join(D, 'nemo_pockets')
REF = 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.2_sw5_l_bfgs_start_sine{}_history.nc'
RUNS = [([os.path.join(D, REF.format(s)) for s in ('', '_seg2', '_seg3')], r'$a = 5$, $w = 0.2$ (reference)', 'k', ':'),
        ([os.path.join(P, 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.5_sw5_l_bfgs_pockets_w0.5_history.nc')],
         r'$a = 5$, $w = 0.5$', 'tab:orange', '-'),
        ([os.path.join(P, 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw1_sw5_l_bfgs_pockets_w1.0_history.nc')],
         r'$a = 5$, $w = 1.0$', 'tab:purple', '-'),
        ([os.path.join(P, 'N64_soft0.0001_energy-warm_Green_Jacobi_eta_0.0625_pfw0.2_sw25_l_bfgs_pockets_w0.2_history.nc')],
         r'$a = 25$, $w = 0.2$', 'tab:brown', '--')]

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
for files, lab, col, ls in RUNS:
    x, f, v, it0 = [], [], [], 0
    for fn in files:
        with netCDF4.Dataset(fn) as d:
            vf = np.array(d['volume_fraction'][:])
        # stress mismatch S per frame (eval_TO3D_objective_split.py; the history may have grown since)
        o = np.genfromtxt(fn.replace('_history.nc', '_objsplit.csv'), delimiter=',', names=True)['stress_mismatch_S']
        vf = vf[:len(o)]
        x.append(it0 + 10 * np.arange(len(o))), f.append(o), v.append(vf)
        it0 += 10 * (len(o) - 1) + 1
    x, f, v = np.concatenate(x), np.concatenate(f), np.concatenate(v)
    a1.plot(x, f, ls, color=col, lw=2, label=lab)
    a2.plot(x, v, ls, color=col, lw=2, label=lab)
    print(f'{lab:30s} iterations ~{x[-1]}  S {f[-1]:.4g}  solid fraction {v[-1]:.3f}')
for S_ch, lab_ch in ((0.5046, 'CH start bi\\_s0'), (0.4286, 'CH start bi\\_s1')):
    a1.axhline(S_ch, color='tab:green', ls='-.', lw=1.5)
    a1.text(1150, S_ch, lab_ch + ' ($a = 5$, $w = 0.2$)', ha='right', va='bottom' if 's0' in lab_ch else 'top',
            fontsize=11, color='tab:green')
a1.set(xlabel='L-BFGS iteration',
       ylabel=r'stress mismatch $S = \sum_G |\sigma_G - \sigma^*_G|^2 / |\sigma^*_G|^2$', ylim=(0, 2.0))
a2.set(xlabel='L-BFGS iteration', ylabel=r'mean $\rho$ (solid fraction)')
for a in (a1, a2):
    a.grid(alpha=0.3), a.set_xlim(0, None)
a1.legend(fontsize=12)
fig.suptitle(r'3D $64^3$ pocket tests, sine start, Green-Jacobi, $\eta = 1/16$ (runs with $w = 0.5, 1.0$ and $a = 25$ '
             r'still running)', fontsize=15)
out = os.path.join(HERE, 'figures', 'TO3D_pockets_history.png')
fig.savefig(out, dpi=130)
print('wrote', out)
