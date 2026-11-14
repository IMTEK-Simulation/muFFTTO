"""Progress of the running local Fracture-Bench Dataset 2 runs (128 x 256): stress per load step, CG per elastic solve
over all staggered iterations (dual x2), damage change per staggered iteration.
    python plot_fb_dataset2_progress.py  ->  figures/fb_dataset2_progress.png"""
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
for P, col, f in (('Green', 'tab:green', 1), ('dual', 'tab:red', 2)):
    st = np.genfromtxt(glob.glob(os.path.join(HERE, 'exp_data', f'FB2_128x256_{P}_*_steps.csv'))[0], delimiter=',', names=True)
    it = np.genfromtxt(glob.glob(os.path.join(HERE, 'exp_data', f'FB2_128x256_{P}_*_iterations.csv'))[0], delimiter=',', names=True)
    ax[0].plot(st['load'] * 1e4, st['stress'] * 377e3, '-o' if P == 'Green' else '--', color=col, ms=4, label=P)
    k = np.arange(1, len(it) + 1)
    ax[1].plot(k, f * it['cg_u'], '-', color=col, lw=1.2, label=P + (r' ($\times 2$)' if f == 2 else '') + f', total {f * it["cg_u"].sum():.0f}')
    ax[2].semilogy(k, it['damage_change'], '-', color=col, lw=1.2, label=P)
    for a in (ax[1], ax[2]):                               # load-step boundaries
        for s in np.flatnonzero(np.diff(it['step'])) + 1.5:
            a.axvline(s, color=col, lw=0.4, alpha=0.3)
    print(P, 'step', int(it['step'][-1]), 'staggered', int(it['staggered_iteration'][-1]), 'last CG', int(it['cg_u'][-1]))
ax[0].set(xlabel=r'strain $\bar\varepsilon_{yy}$ [$10^{-4}$]', ylabel='stress [MPa]', title='macro stress per converged load step')
ax[1].set(xlabel='staggered iteration (all steps)', ylabel='CG per elastic solve', title='elastic CG (dual x2)', ylim=(0, None))
ax[2].set(xlabel='staggered iteration (all steps)', ylabel=r'max $|d - d_{old}|$', title='damage change (crack running)')
for a in ax:
    a.grid(alpha=0.3), a.legend()
fig.suptitle('Fracture-Bench Dataset 2, periodic analogue (local runs in progress): 128 x 256, b/h = 1.28, hard-soft, '
             'pre-crack tip (1.5, 4.2) mm')
out = os.path.join(HERE, 'figures', 'fb_dataset2_progress.png')
fig.savefig(out, dpi=120)
print(out)
