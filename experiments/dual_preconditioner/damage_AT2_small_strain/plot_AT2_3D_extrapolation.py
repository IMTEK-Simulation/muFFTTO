"""3D AT2 (polydisperse ball pores, fixed l = 0.0625, trilinear hexahedra, GPU): elastic CG iterations per solve and
GPU time per elastic solve over the grid N^3 (N = 32, 48, 64, run_AT2_3D_fixed_l.sh), with a power-law fit
a N^b per preconditioner extrapolated to N = 512.

  iterations:  fit to the mean elastic CG iterations per solve
  work:        K applications per solve (dual: 2 per iteration)
  time:        fitted iterations x time per CG iteration, the latter fitted as c N^3 (the operator is matrix-free and
               the FFT is O(N^3 log N); fitted on the measured time_u / cg_u of each run)

The fits use three grids over a factor 2 in N; the extrapolation over a further factor 8 is a trend, not a prediction.

    python plot_AT2_3D_extrapolation.py  -> figures/AT2_3D_extrapolation.png
"""
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TAG = '_polydisperse_pores_l0.0625_3D'
GRIDS = np.array([32, 48, 64])
N_EXTRA = np.array([64, 96, 128, 192, 256, 384, 512])
PRECS = [('Green', 'Green', '#1baf7a', 'o', 1), ('Green_Jacobi', 'Green-Jacobi', '#2a78d6', 's', 1),
         ('dual', 'dual', '#d6402a', '^', 2)]       # (name, label, colour, marker, K applications per iteration)
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5, 'legend.fontsize': 8})


def measured(p):
    """per grid: mean elastic CG iterations per solve, GPU time per elastic CG iteration [s]"""
    its, t_it = [], []
    for N in GRIDS:
        it = np.genfromtxt(os.path.join(HERE, 'exp_data', f'AT2_N{N}_{p}{TAG}_iterations.csv'), delimiter=',',
                           names=True)
        its.append(it['cg_u'].mean())
        t_it.append(it['time_u'].sum() / it['cg_u'].sum())
    return np.array(its), np.array(t_it)


fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
print(f'{"":13s} {"b":>5s} ' + ' '.join(f'{N:>8d}' for N in N_EXTRA[1:]))
for p, label, colour, marker, k_per_it in PRECS:
    its, t_it = measured(p)
    b, log_a = np.polyfit(np.log(GRIDS), np.log(its), 1)          # its = a N^b
    fit_its = np.exp(log_a) * N_EXTRA ** b
    c = np.exp(np.mean(np.log(t_it) - 3 * np.log(GRIDS)))           # time per iteration = c N^3
    for ax, meas, fit in ((axes[0], its, fit_its),
                          (axes[1], k_per_it * its, k_per_it * fit_its),
                          (axes[2], its * t_it, fit_its * c * N_EXTRA ** 3)):
        ax.plot(GRIDS, meas, color=colour, marker=marker, ms=6, lw=2, label=f'{label}: measured')
        ax.plot(N_EXTRA, fit, color=colour, ls='--', lw=1.3, label=f'{label}: fit $\\propto N^{{{b:.2f}}}$')
        ax.annotate(f'{fit[-1]:.3g}', (N_EXTRA[-1], fit[-1]), xytext=(4, 0), textcoords='offset points',
                    va='center', fontsize=8, color='0.25')
    print(f'{label:13s} {b:5.2f} ' + ' '.join(f'{v:8.0f}' for v in fit_its[1:])
          + f'   | time per solve at 512^3: {fit_its[-1] * c * 512 ** 3:.0f} s (GPU, if it fitted)')

for ax, title, ylabel in ((axes[0], 'elastic CG iterations per solve', 'CG iterations'),
                          (axes[1], 'work per solve (dual: 2 K per iteration)', 'K applications'),
                          (axes[2], 'GPU time per elastic solve', 'time [s]')):
    ax.set(xscale='log', yscale='log', xlabel='grid $N$ ($N^3$ voxels)', ylabel=ylabel, title=title)
    ax.set_xticks([32, 64, 128, 256, 512], labels=['32', '64', '128', '256', '512'])
    ax.minorticks_off()
    ax.axvspan(N_EXTRA[0], N_EXTRA[-1], color='0.95', zorder=0)
    ax.grid(alpha=0.3)
axes[0].text(0.97, 0.04, 'shaded: extrapolated', transform=axes[0].transAxes, ha='right', fontsize=8, color='0.35')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right center', frameon=False)
fig.suptitle('3D AT2, polydisperse pores, fixed $l$ = 0.0625 (2 / 3 / 4 voxels at 32 / 48 / 64): measured 32-64, '
             'power-law extrapolation to 512', fontsize=10)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'AT2_3D_extrapolation.png')
fig.savefig(out, dpi=150)
print(f'saved {out}')
