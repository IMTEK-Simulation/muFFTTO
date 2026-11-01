"""
3D: how Green-Jacobi and the dual scale with the grid, depending on the crack setting -- smooth crack (fixed l, l/h grows
with the grid) vs sharp crack (fixed l/h = 2). Elastic CG iterations per solve in the crack steps (load steps with more
than 3 staggered iterations, which dominate the cost); the dual also as cost (2 per iteration, dashed).

    rows: pores, stiff inclusions (10x)
    left    smooth crack, fixed l = 1/16 (32^3, 48^3, 64^3: l/h = 2, 3, 4), full runs (run_AT2_3D_fixed_l.sh; pores only)
    middle  sharp crack, fixed l/h = 2 (32^3, 64^3), full runs (the 64^3 pores dual run stopped inside the break step)
    right   sharp crack, fixed l/h = 2: one cold-start solve on the cut-through damage field, 64^3 -> 96^3
            (diag_3D_frozen.py)

usage:  python plot_AT2_3D_lh_scaling.py       writes figures/AT2_3D_lh_scaling.png
"""
import os

import numpy as np
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GJ, DUAL = ('Green_Jacobi', 'C0', 's', 'Green-Jacobi'), ('dual', 'C3', '^', 'dual')


def crack_mean(n, p, tag):
    d = np.genfromtxt(os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}_polydisperse_pores{tag}_3D_iterations.csv'),
                      delimiter=',', names=True)
    loads = np.unique(d['load'])
    calm = np.isin(d['load'], loads[np.array([(d['load'] == e).sum() for e in loads]) <= 3])
    return d['cg_u'][~calm].mean()


def frozen(case, n, p):
    d = np.genfromtxt(os.path.join(HERE, 'exp_data', f'diag_3D_frozen_{case}_N{n}.csv'), delimiter=',', names=True,
                      dtype=None, encoding=None)
    return float(d['cg_its'][(d['state'] == 'broken') & (d['preconditioner'] == p)][0])


SUB_RUNS = 'full runs: mean per elastic solve in the crack steps'
SUB_FROZEN = 'frozen cut-through field: one solve'
panels = {(0, 0): ('pores, smooth crack: fixed l = 1/16\n(l/h = 2, 3, 4 on 32$^3$, 48$^3$, 64$^3$)', SUB_RUNS,
                   {p: [(n, crack_mean(n, p, '_l0.0625')) for n in (32, 48, 64)] for p in (GJ[0], DUAL[0])}),
          (0, 1): ('pores, sharp crack: fixed l/h = 2', SUB_RUNS,
                   {p: [(32, crack_mean(32, p, '_l0.0625')), (64, crack_mean(64, p, '_l0.03125'))]
                    for p in (GJ[0], DUAL[0])}),
          (0, 2): ('pores, sharp crack: fixed l/h = 2', SUB_FROZEN,
                   {p: [(n, frozen('pores', n, p)) for n in (64, 96)] for p in (GJ[0], DUAL[0])}),
          (1, 0): ('stiff inclusions, smooth crack', 'no fixed-l runs', None),
          (1, 1): ('stiff inclusions, sharp crack: fixed l/h = 2', SUB_RUNS,
                   {p: [(32, crack_mean(32, p, '_incl10_l0.0625')), (64, crack_mean(64, p, '_incl10_l0.03125'))]
                    for p in (GJ[0], DUAL[0])}),
          (1, 2): ('stiff inclusions, sharp crack: fixed l/h = 2', SUB_FROZEN,
                   {p: [(n, frozen('incl', n, p)) for n in (64, 96)] for p in (GJ[0], DUAL[0])})}

fig, axes = plt.subplots(2, 3, figsize=(18, 11), layout='constrained', dpi=150)
handles = {}
for (row, col), (title, subtitle, data) in panels.items():
    ax = axes[row, col]
    ax.set_title(f'{title}\n{subtitle}', fontsize=12.5)
    if data is None:
        ax.axis('off')
        continue
    grids = sorted({n for pts in data.values() for n, _ in pts})
    for p, c, m, label in (GJ, DUAL):
        n, it = np.array(data[p]).T
        handles.setdefault(label, ax.plot(n, it, '-', marker=m, ms=8, lw=2.2, color=c, label=label)[0])
        for x, y in zip(n, it):
            ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points', xytext=(7, -14 if p == 'dual' else 7),
                        fontsize=11, color=c)
        if p == 'dual':
            handles.setdefault('dual, cost (2x)', ax.plot(n, 2 * it, '--', marker=m, ms=8, lw=2, color=c, alpha=0.45,
                                                          mfc='white', label='dual, cost (2x)')[0])
    ax.set_xticks(grids), ax.set_xticklabels([f'{g}$^3$' for g in grids])
    ax.set_xlim(grids[0] - 8, grids[-1] + 8), ax.set_ylim(0, None)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, integer=True))
    ax.set_xlabel('grid', fontsize=12), ax.set_ylabel('CG iterations per elastic solve', fontsize=12)
    ax.tick_params(labelsize=12), ax.grid(True, color='0.9')
fig.legend(handles=list(handles.values()), loc='outside lower center', ncol=3, fontsize=13)
out = os.path.join(HERE, 'figures', 'AT2_3D_lh_scaling.png')
fig.savefig(out)
print(f'wrote {out}')
