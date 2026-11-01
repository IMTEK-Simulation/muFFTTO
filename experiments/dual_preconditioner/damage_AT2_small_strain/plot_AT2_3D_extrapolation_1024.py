"""
3D: power-law extrapolation a N^b of the elastic CG iterations per solve in the crack steps to N = 1024, per crack
setting (the series of plot_AT2_3D_lh_scaling.py, full runs):

    pores, smooth crack (fixed l = 1/16; 32^3, 48^3, 64^3)
    pores, sharp crack (fixed l/h = 2; 32^3, 64^3 -- the 64^3 dual run stopped inside the break step, no 64^3 Green)
    stiff inclusions, sharp crack (fixed l/h = 2; 32^3, 64^3)

The dual also as work (2 K applications per iteration, dashed). Fits on two or three grids over a factor 2 in N,
extrapolated over a further factor 16: a trend, not a prediction (and a 1024^3 run does not fit any single machine).

usage:  python plot_AT2_3D_extrapolation_1024.py       writes figures/AT2_3D_extrapolation_1024.png, prints the fits
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'C2', 'o', 'Green', 1), ('Green_Jacobi', 'C0', 's', 'Green-Jacobi', 1), ('dual', 'C3', '^', 'dual', 2)]
N_EXTRA = np.array([32, 1024])
TICKS = [32, 64, 128, 256, 512, 1024]


def crack_mean(n, p, tag):
    f = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}_polydisperse_pores{tag}_3D_iterations.csv')
    if not os.path.exists(f):
        return None
    d = np.genfromtxt(f, delimiter=',', names=True)
    loads = np.unique(d['load'])
    calm = np.isin(d['load'], loads[np.array([(d['load'] == e).sum() for e in loads]) <= 3])
    return d['cg_u'][~calm].mean()


SERIES = [('pores, smooth crack\nfixed l = 1/16 (l/h = 2, 3, 4)', [(32, '_l0.0625'), (48, '_l0.0625'), (64, '_l0.0625')]),
          ('pores, sharp crack\nfixed l/h = 2', [(32, '_l0.0625'), (64, '_l0.03125')]),
          ('stiff inclusions, sharp crack\nfixed l/h = 2', [(32, '_incl10_l0.0625'), (64, '_incl10_l0.03125')])]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.8), layout='constrained', dpi=150)
handles = {}
for ax, (title, runs) in zip(axes, SERIES):
    print(title.replace('\n', ', '))
    for p, c, m, label, work in PRECS:
        pts = [(n, crack_mean(n, p, tag)) for n, tag in runs]
        pts = [(n, it) for n, it in pts if it is not None]
        if len(pts) < 2:
            continue
        n, it = np.array(pts).T
        b, log_a = np.polyfit(np.log(n), np.log(it), 1)
        fit = np.exp(log_a) * N_EXTRA.astype(float) ** b
        for scale, ls, alpha, lab in ((1, '-', 1.0, label), (2, '--', 0.45, 'dual, work (2x)')) if p == 'dual' \
                else ((1, '-', 1.0, label),):
            h, = ax.plot(n, scale * it, linestyle='none', marker=m, ms=9, color=c, alpha=alpha, label=lab,
                         mfc=c if scale == 1 else 'white')
            ax.plot(N_EXTRA, scale * fit, linestyle=':' if scale == 1 else (0, (2, 4)), lw=2, color=c, alpha=alpha)
            handles.setdefault(lab, h)
            ax.annotate(f'{scale * fit[-1]:.0f}', (1024, scale * fit[-1]), textcoords='offset points',
                        xytext=(6, 0), va='center', fontsize=11, color=c, alpha=max(alpha, 0.7))
        print(f'  {label:13s} its ~ N^{b:.2f}   at 1024: {fit[-1]:.0f}' + (f' (work {2 * fit[-1]:.0f})' if work == 2 else ''))
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xticks(TICKS), ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}$^3$'))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(28, 1500), ax.set_ylim(20, 12000)
    ax.set_title(f'{title}\nmarkers: full runs (crack steps); dotted: fit a N$^b$', fontsize=12.5)
    ax.set_xlabel('grid', fontsize=12), ax.set_ylabel('CG iterations per elastic solve', fontsize=12)
    ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
fig.legend(handles=list(handles.values()), loc='outside lower center', ncol=4, fontsize=13)
out = os.path.join(HERE, 'figures', 'AT2_3D_extrapolation_1024.png')
fig.savefig(out)
print(f'wrote {out}')
