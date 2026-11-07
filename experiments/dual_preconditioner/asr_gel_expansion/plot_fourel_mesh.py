"""Mesh-size study of the Fourel-model runs (fourel_nonlocal_damage.py; laptop and NEMO via sync_nemo_fourel.sh),
same format as ../phase_field_fatigue/figures/FAT_mesh_cg_vs_lh.png.

Fixed l_c = 0.25 mm, cell 35 mm (2D) / 35 mm cube (3D): l_c/h = N * 0.25 / 35 (64 -> 0.46, 128 -> 0.91,
256 -> 1.83, 512 -> 3.66, 1024 -> 7.31). LOADING=free, K_RESIDUAL=1e-4, one spot per solve.
Every grid is compared on a common eps_ASR window [0, eps_end(N)] = the smallest eps_ASR reached by all its runs
(env EPS_CAP caps it, e.g. EPS_CAP=0.01).
* before cracking: elastic solves before the first damage spot
* during cracking: solves with damage (one per spot) inside the window
* total work: all iterations inside the window (Newton: incl. unit solves for C_eff), dual x 2
Cost unit = one Green-PCG iteration = one MS fixed-point iteration; dual iterations count 2.
Open markers: l_c/h < 2 (under-resolved, grey band).

figures/FOUREL_mesh_cg_vs_lh.png; prints the table.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
L_C, CELL = 0.25, 35.0
eps_cap = float(os.environ.get('EPS_CAP', 1.0))
SOLVERS = [('MS', '_tolms1e-06', 'MS basic scheme, tol 1e-6', 'grey', 'D', '-', 1),
           ('MS', '', 'MS basic scheme, tol 1e-4', 'black', 'x', '--', 1),
           ('Green', '', 'Newton + Green', 'tab:green', 'o', '-', 1),
           ('Green_Jacobi', '', 'Newton + Green-Jacobi', 'tab:blue', 's', '-', 1),
           ('dual', '', 'Newton + dual', 'tab:red', '^', '-', 2)]
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 11})


def load(D, N, P, tag):
    base = os.path.join(here, 'exp_data', f'FOUREL{"3D" if D == "3D" else ""}_N{N}_{P}_free_k0.0001{tag}')
    if not (os.path.exists(base + '_solves.csv') and os.path.exists(base + '_steps.npz')):
        return None
    st = np.load(base + '_steps.npz', allow_pickle=True)
    sv = np.atleast_1d(np.genfromtxt(base + '_solves.csv', delimiter=',', names=True, dtype=None, encoding=None))
    sv = sv[sv['step'] < len(st['eps_asr'])]
    return dict(sv=sv, eps=st['eps_asr'], unit=st['unit_its'], spots=st['spots'])


rows = []
for D, NS in (('2D', (64, 128, 256, 512, 1024)), ('3D', (64, 128, 256))):
    for N in NS:
        runs = {(P, tag): load(D, N, P, tag) for P, tag, *_ in SOLVERS}
        runs = {k: v for k, v in runs.items() if v is not None and len(v['eps'])}
        if not runs:
            continue
        ends = {k: v['eps'][-1] for k, v in runs.items()}
        eps_end = min(min(ends.values()), eps_cap)
        for (P, tag), r in runs.items():
            w = [s[-1] for s in SOLVERS if s[0] == P and s[1] == tag][0]
            sv = r['sv']
            win = sv['eps_asr'] <= eps_end + 1e-12
            pre = win & (sv['spots_total'] == 0)
            dur = win & ~pre
            lin = sv['linear_its'] + sv['unit_its']
            per = lambda m: w * lin[m].sum() / m.sum() if m.sum() else np.nan
            e = sv['energy_error'][win]
            e = e[~np.isnan(e)]
            rows.append(dict(D=D, N=N, P=P, tag=tag, lh=N * L_C / CELL, eps_end=eps_end, solves=int(win.sum()),
                             pre=per(pre), dur=per(dur),
                             work=w * (lin[win].sum() + r['unit'][r['eps'] <= eps_end + 1e-12].sum()),
                             err=np.median(e) if e.size else np.nan, full=ends[(P, tag)] >= max(ends.values()) - 1e-12))
print(f'{"D":>2} {"N":>5} {"l/h":>5} {"solver":>22} {"eps_end%":>8} {"solves":>6} {"per solve pre":>13} '
      f'{"during":>7} {"work":>10} {"err":>8}')
for r in rows:
    print(f'{r["D"]:>2} {r["N"]:5d} {r["lh"]:5.2f} {r["P"] + r["tag"]:>22} {100 * r["eps_end"]:8.3f} {r["solves"]:6d} '
          f'{r["pre"]:13.1f} {r["dur"]:7.1f} {r["work"]:10.0f} {r["err"]:8.1e}' + (' under-resolved' if r['lh'] < 2 else ''))

dims = [D for D in ('2D', '3D') if any(r['D'] == D for r in rows)]
fig, axes = plt.subplots(len(dims), 3, figsize=(16, 5 * len(dims) + 0.8), squeeze=False)
for k, D in enumerate(dims):
    for P, tag, label, c, m, ls, w in SOLVERS:
        rr = sorted([r for r in rows if r['D'] == D and r['P'] == P and r['tag'] == tag], key=lambda r: r['lh'])
        if not rr:
            continue
        lh = np.array([r['lh'] for r in rr])
        for j, key in enumerate(('pre', 'dur')):
            y = np.array([r[key] for r in rr])
            axes[k, j].plot(lh, y, ls=ls, color=c, lw=2)
            for x_, y_, r in zip(lh, y, rr):
                axes[k, j].plot(x_, y_, m, color=c, ms=9, mfc='white' if x_ < 2 else c)
        y = np.array([r['work'] / 1e3 for r in rr])
        axes[k, 2].plot(lh, y, ls=ls, color=c, lw=2, label=label + (' (x 2)' if w == 2 else ''))
        for x_, y_, r in zip(lh, y, rr):
            axes[k, 2].plot(x_, y_, m, color=c, ms=9, mfc='white' if x_ < 2 else c)
    ticks = sorted({r['lh'] for r in rows if r['D'] == D})
    win = {r['N']: r['eps_end'] for r in rows if r['D'] == D}
    axes[k, 0].set_title(f'{D}: before cracking', fontsize=13)
    axes[k, 1].set_title(f'{D}: during cracking (one solve per spot)', fontsize=13)
    axes[k, 2].set_title(f'{D}: total work up to the common ' + r'$\varepsilon_{ASR}$', fontsize=13)
    for j in range(3):
        ax = axes[k, j]
        ax.set_xscale('log', base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.2g}\n({N}, {100 * win[N]:.2g} %)' for t, N in zip(ticks, sorted(win))], fontsize=10)
        ax.minorticks_off()
        ax.set_xlabel(r'$l_c / h$   (grid N, $\varepsilon_{ASR}$ window)')
        ax.set_ylim(0, None)
        ax.axvspan(min(ticks) / 1.3, 2, color='0.9', zorder=0)
        ax.set_xlim(min(ticks) / 1.3, max(max(ticks), 2) * 1.3)
        ax.grid(alpha=0.3)
    axes[k, 0].set_ylabel('cost per elastic solve\n[Green-iteration units]')
    axes[k, 1].set_ylabel('cost per elastic solve\n[Green-iteration units]')
    axes[k, 2].set_ylabel('total work [10$^3$ units] (dual x 2)')
h, l = axes[0, 2].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=11,
           title='grey band, open markers: l_c/h < 2 (under-resolved)',
           title_fontsize=11)
fig.tight_layout(rect=(0, 0.09, 1, 1))
out = os.path.join(here, 'figures', 'FOUREL_mesh_cg_vs_lh.png')
fig.savefig(out, dpi=150)
print('->', out)
