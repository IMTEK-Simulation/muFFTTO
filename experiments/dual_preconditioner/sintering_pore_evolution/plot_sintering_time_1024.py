"""Wall time of the elastic solves of the 1024^2 sintering runs (NEMO, one run per L40S GPU, 603 solves per run):
total time and time per CG iteration, Green-Jacobi vs dual, W = 10 nm and W = 4 nm. No Green run at 1024^2.
    python plot_sintering_time_1024.py  ->  figures/sintering_time_1024.{png,pdf}
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [('W10_P50u_k0.0001_W10', r'$W = 10$ nm'), ('W4_P50u_k0.0001', r'$W = 4$ nm')]
ONLY_W4 = os.environ.get('ONLY_W4') == '1'                  # only W = 4 nm -> sintering_time_1024_W4
if ONLY_W4:
    CASES = [(CASES[1][0], r'$l = 4$ nm')]                  # talk notation: l for the interface width
PRE = [('Green_Jacobi', 'Green-Jacobi', 'tab:blue'), ('dual', 'dual', 'tab:red')]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 14})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7 if ONLY_W4 else 10, 4.2), layout='constrained')
x = np.arange(len(CASES))
wd = 0.36
for j, (P, lab, col) in enumerate(PRE):
    tot, per_it, cg = [], [], []
    for pat, _ in CASES:
        z = np.load(os.path.join(HERE, 'exp_data', f'SINT_N1024_{P}_{pat}_steps.npz'), allow_pickle=True)
        t = float(np.nansum(z['time_mech']))
        its = float(np.nansum(z['cg']))
        tot.append(t / 60), per_it.append(1e3 * t / its), cg.append(float(np.nanmean(z['cg_mean'])))
    b = a1.bar(x + (j - 0.5) * wd, tot, wd, color=col, label=lab)
    for xi, (h, c) in enumerate(zip(tot, cg)):
        a1.text(xi + (j - 0.5) * wd, h, f'{h:.0f} min\n{c:.0f} CG', ha='center', va='bottom', fontsize=11)
    a2.bar(x + (j - 0.5) * wd, per_it, wd, color=col, label=lab)
    for xi, h in enumerate(per_it):
        a2.text(xi + (j - 0.5) * wd, h, f'{h:.1f} ms', ha='center', va='bottom', fontsize=11)
    print(lab, 'total min', [round(v, 1) for v in tot], 'ms/CG', [round(v, 1) for v in per_it], 'CG/solve', [round(v) for v in cg])
for a in (a1, a2):
    a.set_xticks(x), a.set_xticklabels([c[1] for c in CASES]), a.grid(alpha=0.3, axis='y')
a1.set(ylabel='elastic solves, total wall time [min]', ylim=(0, 90 if ONLY_W4 else 70))
a2.set(ylabel='time per CG iteration [ms]', ylim=(0, 40))
a1.legend(loc='upper left')
fig.suptitle(r'Sintering $1024^2$: elastic solves (603 per run), one L40S GPU', fontsize=15)
out = os.path.join(HERE, 'figures', 'sintering_time_1024' + ('_W4' if ONLY_W4 else ''))
for ext in ('png', 'pdf'):
    fig.savefig(f'{out}.{ext}', dpi=150)
print('wrote', out)
