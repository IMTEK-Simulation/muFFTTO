"""Porous AT2 problem with the crack sharpening with the grid: l = (l/h) h at fixed l/h = 2 and 4, mean CG per elasticity
solve over the whole run against the grid. Green / Green-Jacobi / dual (counted x2), measured runs only.
    python plot_AT2_sharp_crack_mesh.py  ->  figures/AT2_sharp_crack_mesh_lh{2,4}.{png,pdf}
"""
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
GRIDS = [32, 64, 128, 256, 512]
PRE = [('Green', 'Green', 'tab:green', 'o', 1), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's', 1),
       ('dual', r'dual ($\times 2$)', 'tab:red', '^', 2)]


def mean_cg(N, P, lh):
    for tag in ('_lh', '_smooth', ''):                      # the same setting under different study names
        f = os.path.join(HERE, 'exp_data', f'AT2_N{N}_{P}_polydisperse_pores_l{lh / N:.8g}{tag}_iterations.csv')
        if os.path.exists(f):
            v = [float(r['cg_u']) for r in csv.DictReader(open(f)) if r['cg_u'] not in ('', 'nan')]
            return sum(v) / len(v), len(v), tag
    return None


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
for lh in (2, 4):
    fig = plt.figure(figsize=(4.4, 3.7))
    ax = fig.add_axes([0.16, 0.16, 0.76, 0.74])
    for P, lab, col, mk, fac in PRE:
        pts = [(N, mean_cg(N, P, lh)) for N in GRIDS]
        pts = [(N, fac * m[0]) for N, m in pts if m]
        ax.plot(*zip(*pts), '-' + mk, color=col, lw=2.2, ms=6, clip_on=False, zorder=5, label=lab)
        print(f'l/h={lh}', lab, [(n, round(v)) for n, v in pts])
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
    ax.set_xlim(32, 512), ax.set_ylim(0, 450)
    ax.set(xlabel='grid size', ylabel='CG iterations per solve', title=f'$l = {lh}h$')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, handlelength=2.6, framealpha=0.95)
    out = os.path.join(HERE, 'figures', f'AT2_sharp_crack_mesh_lh{lh}')
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
    plt.close(fig)
