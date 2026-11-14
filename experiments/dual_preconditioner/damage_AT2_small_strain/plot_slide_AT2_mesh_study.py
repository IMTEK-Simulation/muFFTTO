"""Mesh studies for the fracture slides, l = 1/32 fixed, mean CG per elasticity solve over the whole run against the grid,
Green / Green-Jacobi / dual (counted x2), measured runs only:
  porous: polydisperse pores (the 1024^2 video problem): Green-Jacobi up to 256^2, no Green at 512^2
  ch:     Cahn-Hilliard two-phase, continuous cracking (schneider2025_notch, _ch; not the gc1.7407 single-snap case), 64^2-512^2
    NO_DUAL=1 (env): porous case only, Green and Green-Jacobi -> slide_AT2_mesh_study_nodual
    python plot_slide_AT2_mesh_study.py  ->  figures/slide_AT2_mesh_study{,_ch}.{png,pdf} (+ copy to the talk figures/)
"""
import csv
import os
import shutil

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
TALK = os.path.expanduser('~/Projects/Talks/Slides-Green-Jacobi-topology-optimization-NANTES/figures')
GRIDS = [32, 64, 128, 256, 512, 1024]
PRE = [('Green', 'Green', 'tab:green', 'o', 1), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's', 1),
       ('dual', r'dual ($\times 2$)', 'tab:red', '^', 2)]
CASES = [('exp_data/AT2_N{}_{}_polydisperse_pores_l0.03125_iterations.csv', 'slide_AT2_mesh_study'),
         ('schneider2025_notch/exp_data/AT2_N{}_{}_incl0.1_l0.03125_k0.0001_ch_iterations.csv', 'slide_AT2_mesh_study_ch')]
NO_DUAL = os.environ.get('NO_DUAL') == '1'
if NO_DUAL:
    PRE, CASES = PRE[:2], [(CASES[0][0], 'slide_AT2_mesh_study_nodual')]
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
for pat, name in CASES:
    fig = plt.figure(figsize=(4.4, 3.7))
    ax = fig.add_axes([0.16, 0.16, 0.76, 0.80])
    for P, lab, col, mk, fac in PRE:
        pts = []
        for N in GRIDS:
            f = os.path.join(HERE, pat.format(N, P))
            if os.path.exists(f):
                v = [float(r['cg_u']) for r in csv.DictReader(open(f)) if r['cg_u'] not in ('', 'nan')]
                pts.append((N, fac * sum(v) / len(v)))
        ax.plot(*zip(*pts), '-' + mk, color=col, lw=2.2, ms=6, clip_on=False, zorder=5, label=lab)
        print(name, lab, [(n, round(v)) for n, v in pts])
    ax.set_xscale('log', base=2)
    ax.set_xticks(GRIDS), ax.set_xticklabels([f'{n}$^2$' for n in GRIDS]), ax.minorticks_off()
    ax.set_xlim(32, 1024), ax.set_ylim(0, 750)
    ax.set(xlabel='grid size', ylabel='CG iterations per solve')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', fontsize=12, handlelength=2.6, framealpha=0.95)
    out = os.path.join(HERE, 'figures', name)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{out}.{ext}', dpi=200)
        shutil.copy(f'{out}.{ext}', TALK)
    plt.close(fig)
