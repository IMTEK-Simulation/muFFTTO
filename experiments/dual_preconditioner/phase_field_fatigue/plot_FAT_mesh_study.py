"""Mesh-size study of the NEMO fatigue runs (sbatch_fatigue_gpu.sh, run_FAT_nemo2D_local.sh), see README.md.

Fixed l = L/64 (Carrara parameters, glass beads), grids 2D 32^2 ... 1024^2 and 3D 32^3 ... 256^3, l/h = N/64.
Restart segments (_r1, _r2) are joined (a later segment replaces the steps it repeats).

figures/FAT_mesh_cg_vs_lh.png       mean CG per elastic solve before / during growth vs l/h, 2D and 3D, + total work
figures/FAT_mesh_crack_growth.png   crack length vs cycle, 2D and 3D, all grids (dual runs, else any)
prints the summary table (l/h < 2 flagged as under-resolved).
"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
L_OVER_CELL = 1.0 / 64.0
L_MM = 12.8
PRECS = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
         ('dual', 'dual', 'tab:red', '^')]
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 12})


def load(D, N, P):
    files = glob.glob(os.path.join(here, 'exp_data', f'FAT_N{N}_{P}_incl*_nemo{D}*_precrack_*_cycles.csv'))
    files = [f for f in files if re.search(rf'_nemo{D}(_r\d+)?_precrack', f)]
    if not files:
        return None
    seg = lambda f: int(m[1]) if (m := re.search(r'_nemo\dD_r(\d+)_precrack', f)) else 0
    files.sort(key=seg)
    parts = [np.genfromtxt(f, delimiter=',', names=True, ndmin=1) for f in files]
    parts = [p for p in parts if p.size]
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = np.concatenate([out[out['step'] < p['step'].min()], p])
    return out


def phases(d, N):
    grown = d['crack_length'] > d['crack_length'][0] + 2.5 / N
    i_init = int(np.argmax(grown)) if grown.any() else len(d)
    broken = np.r_[False, d['sigma'][1:] < 0.7 * d['sigma'][:-1]]
    i_break = int(np.argmax(broken)) if broken.any() else len(d)
    return slice(0, i_init), slice(i_init, i_break), i_init, i_break


rows = []
for D, NS in (('2D', (32, 64, 128, 256, 512, 1024)), ('3D', (32, 64, 128, 256))):
    for N in NS:
        for P, _, _, _ in PRECS:
            d = load(D, N, P)
            if d is None:
                continue
            pre, grow, i_init, i_break = phases(d, N)
            cg, its = d['cg_u_step'], d['staggered_its']
            per = lambda s: cg[s].sum() / its[s].sum() if its[s].sum() else np.nan
            finished = i_break < len(d)
            rows.append(dict(D=D, N=N, P=P, lh=N * L_OVER_CELL, cycles=len(d), init=i_init + 1, brk=i_break + 1,
                             pre=per(pre), grow=per(grow), work=(2 if P == 'dual' else 1) * cg.sum(),
                             solves=int(its.sum()), finished=finished, t=float(d['time_u_step'].sum()),
                             dadn=(d['crack_length'][max(i_break - 1, i_init)] - d['crack_length'][min(i_init, len(d) - 1)])
                             / max(i_break - 1 - i_init, 1) * L_MM * 1e3 if i_init < len(d) else np.nan))
print(f'{"D":>2} {"N":>5} {"l/h":>5} {"P":>13} {"cycles":>6} {"init":>5} {"break":>5} {"CG/solve pre":>12} '
      f'{"growth":>7} {"work":>10} {"t_u[h]":>7} {"da/dN um":>8}')
for r in rows:
    flag = ' under-resolved' if r['lh'] < 2 else ''
    flag += '' if r['finished'] else ' (running / not broken)'
    print(f'{r["D"]:>2} {r["N"]:5d} {r["lh"]:5.2g} {r["P"]:>13} {r["cycles"]:6d} {r["init"]:5d} {r["brk"]:5d} '
          f'{r["pre"]:12.1f} {r["grow"]:7.1f} {r["work"]:10.0f} {r["t"] / 3600:7.2f} {r["dadn"]:8.1f}{flag}')

fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
for k, D in enumerate(('2D', '3D')):
    for P, label, c, m in PRECS:
        rr = [r for r in rows if r['D'] == D and r['P'] == P]
        if not rr:
            continue
        lh = np.array([r['lh'] for r in rr])
        for j, key in enumerate(('pre', 'grow')):
            y = np.array([r[key] for r in rr])
            axes[k, j].plot(lh, y, '-', color=c, lw=2, label=label)
            for x_, y_, r in zip(lh, y, rr):         # hollow markers: under-resolved (l/h < 2) or unfinished
                hollow = x_ < 2 or not r['finished']
                axes[k, j].plot(x_, y_, m, color=c, ms=9, mfc='white' if hollow else c)
        axes[k, 2].plot(lh, [r['work'] / 1e3 for r in rr], '-' + m, color=c, lw=2, ms=8,
                        label=label + (' (x 2)' if P == 'dual' else ''))
    axes[k, 0].set_title(f'{D}: before growth', fontsize=13)
    axes[k, 1].set_title(f'{D}: during growth', fontsize=13)
    axes[k, 2].set_title(f'{D}: total elastic work', fontsize=13)
    for j in range(3):
        ax = axes[k, j]
        ax.set_xscale('log', base=2)
        ticks = sorted({r['lh'] for r in rows if r['D'] == D})
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:g}' for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel('l / h  (grid N = 64 l/h)')
        ax.set_ylim(0, None)
        ax.axvspan(min(ticks + [0.5]) / 1.3, 2, color='0.9', zorder=0)
        ax.grid(alpha=0.3)
    axes[k, 0].set_ylabel('mean CG iterations per elastic solve')
    axes[k, 1].set_ylabel('mean CG iterations per elastic solve')
    axes[k, 2].set_ylabel('CG iterations x 1000 (dual x 2)')
axes[0, 0].legend(title='grey band: l/h < 2 (under-resolved)', loc='upper left', fontsize=11, title_fontsize=11)
axes[0, 2].legend(loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(here, 'figures', 'FAT_mesh_cg_vs_lh.png'), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
styles = ['-', '--', '-.', ':', (0, (5, 1)), (0, (1, 1))]
for ax, (D, NS) in zip(axes, (('2D', (32, 64, 128, 256, 512, 1024)), ('3D', (32, 64, 128, 256)))):
    for i, N in enumerate(NS):
        d = next((load(D, N, P) for P in ('dual', 'Green_Jacobi', 'Green') if load(D, N, P) is not None), None)
        if d is None:
            continue
        ax.plot(d['step'] + 1, d['crack_length'] * L_MM, ls=styles[i % len(styles)],
                color=plt.cm.Greys(0.35 + 0.65 * i / max(len(NS) - 1, 1)), lw=2, label=f'{N} (l/h = {N / 64:g})')
    ax.set_title(f'{D}: crack length (d > 0.95)', fontsize=13)
    ax.set_xlabel('cycle N')
    ax.set_ylabel('crack length [mm] (cell 12.8 mm)')
    ax.set_xlim(left=0)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.legend(title='grid', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(here, 'figures', 'FAT_mesh_crack_growth.png'), dpi=150)
print('written figures/FAT_mesh_cg_vs_lh.png, figures/FAT_mesh_crack_growth.png')
