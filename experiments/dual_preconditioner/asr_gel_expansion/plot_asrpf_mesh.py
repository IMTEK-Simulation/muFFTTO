"""Mesh-size study of asr_pf_fourel.py (format of ../phase_field_fatigue/plot_FAT_mesh_study.py).

Fixed l = b = 0.5 mm on the 35 mm cell, l/h = N / 70: 64^2 / 128^2 / 256^2 -> 0.91 / 1.83 / 3.66 (2D, laptop:
exp_data/), 3D 64^3 ... from NEMO (exp_data/nemo_pf/, filled in when they exist; restart segments write the same
files). Compared on a common eps_ASR window per variant (A: eps <= EPS_A = 0.5 %, B: eps <= EPS_B = 1.0 %), so every
grid / preconditioner covers the same load path. 'before cracking' / 'during cracking' = load steps ending with
max d (PF-CZM: 1 - omega(d)) < 0.9 / >= 0.9.

figures/ASRpf_mesh_cg_vs_lh_A.png, figures/ASRpf_mesh_cg_vs_lh_B.png: mean CG per elastic solve before / during
cracking and the total elastic work (all main + unit solves, dual x 2) vs l/h; rows 2D, 3D; grey band l/h < 2
(under-resolved) with open markers. Prints the table.
    python plot_asrpf_mesh.py        env: EPS_A (0.005), EPS_B (0.01)
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
CELL_MM, L_MM = 35.0, 0.5
PRECS = [('Green', 'Green', 'tab:green', 'o'), ('Green_Jacobi', 'Green-Jacobi', 'tab:blue', 's'),
         ('dual', 'dual', 'tab:red', '^')]
VARIANTS = {'A': ('fourel', float(os.environ.get('EPS_A', 0.005)), 'A: AT1 Fourel-matched'),
            'B': ('czm', float(os.environ.get('EPS_B', 0.01)), 'B: PF-CZM real concrete')}
GRIDS = {'2D': (64, 128, 256, 512, 1024), '3D': (32, 64, 128, 256)}
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 12})


def load(variant, D, N, P):
    name = f'ASRpf_{variant}_N{N}_{P}' if D == '2D' else f'ASRpf_{variant}_3D_N{N}_{P}'
    for d in ('exp_data', 'exp_data/nemo_pf'):
        base = os.path.join(here, d, name)
        if os.path.exists(base + '_steps.npz'):
            r = dict(np.load(base + '_steps.npz', allow_pickle=True))
            key = 'loss' if glob.glob(base + '_damage/loss_*') else 'step'
            files = sorted(glob.glob(base + f'_damage/{key}_*[0-9].npy') + glob.glob(base + f'_damage/{key}_*_midz.npy'))
            r['field_max'] = np.array([float(np.load(f).max()) for f in files[:len(r['eps_asr'])]])
            return r
    return None


for vkey, (variant, eps_win, vtitle) in VARIANTS.items():
    rows = []
    for D, NS in GRIDS.items():
        for N in NS:
            for P, _, _, _ in PRECS:
                r = load(variant, D, N, P)
                if r is None:
                    continue
                n = int(np.sum(r['eps_asr'] <= eps_win + 1e-12))
                if n == 0:
                    continue
                crack = r['field_max'][:n] >= 0.9
                cg, its = r['cg_main_sum'][:n], r['staggered'][:n]
                per = lambda s: cg[s].sum() / its[s].sum() if its[s].sum() else np.nan
                rows.append(dict(D=D, N=N, P=P, lh=N * L_MM / CELL_MM, n=n, finished=r['eps_asr'][n - 1] >= eps_win - 1e-12,
                                 pre=per(~crack), grow=per(crack), its=int(its.sum()),
                                 work=(2 if P == 'dual' else 1) * float(np.sum(r['cg_all_sum'][:n])),
                                 t=float(np.sum(r['time_step'][:n]))))
    print(f'\n{vtitle}, eps_ASR <= {100 * eps_win:g} %')
    print(f'{"D":>2} {"N":>5} {"l/h":>5} {"P":>13} {"steps":>5} {"stagg.its":>9} {"CG/solve pre":>12} {"crack":>7} '
          f'{"work (x2 dual)":>14} {"t [s]":>8}')
    for r in rows:
        flag = (' under-resolved' if r['lh'] < 2 else '') + ('' if r['finished'] else ' (window not reached)')
        print(f'{r["D"]:>2} {r["N"]:5d} {r["lh"]:5.2f} {r["P"]:>13} {r["n"]:5d} {r["its"]:9d} {r["pre"]:12.1f} '
              f'{r["grow"]:7.1f} {r["work"]:14.0f} {r["t"]:8.0f}{flag}')

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for k, D in enumerate(GRIDS):
        ticks = sorted({r['lh'] for r in rows if r['D'] == D})
        for P, label, c, m in PRECS:
            rr = [r for r in rows if r['D'] == D and r['P'] == P]
            if not rr:
                continue
            lh = np.array([r['lh'] for r in rr])
            for j, key in enumerate(('pre', 'grow')):
                y = np.array([r[key] for r in rr])
                axes[k, j].plot(lh, y, '-', color=c, lw=2, label=label)
                for x_, y_, r in zip(lh, y, rr):     # hollow markers: under-resolved (l/h < 2) or window not reached
                    axes[k, j].plot(x_, y_, m, color=c, ms=9, mfc='white' if (x_ < 2 or not r['finished']) else c)
            axes[k, 2].plot(lh, [r['work'] / 1e3 for r in rr], '-', color=c, lw=2,
                            label=label + (' (x 2)' if P == 'dual' else ''))
            for x_, r in zip(lh, rr):                # hollow: window not reached (partial work)
                axes[k, 2].plot(x_, r['work'] / 1e3, m, color=c, ms=8, mfc=c if r['finished'] else 'white')
        axes[k, 0].set_title(f'{D}: before cracking', fontsize=13)
        axes[k, 1].set_title(f'{D}: during cracking', fontsize=13)
        axes[k, 2].set_title(f'{D}: total elastic work, eps_ASR <= {100 * eps_win:g} %', fontsize=13)
        for j in range(3):
            ax = axes[k, j]
            if ticks:
                ax.set_xscale('log', base=2)
                ax.set_xticks(ticks)
                ax.set_xticklabels([f'{t:.2g}' for t in ticks])
                ax.minorticks_off()
                ax.axvspan(min(ticks + [0.5]) / 1.3, 2, color='0.9', zorder=0)
            else:
                ax.text(0.5, 0.5, f'{D}: no runs yet (NEMO)', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
            ax.set_xlabel('l / h  (l = b = 0.5 mm, h = 35 mm / N)')
            ax.set_ylim(0, None)
            ax.grid(alpha=0.3)
        axes[k, 0].set_ylabel('mean CG iterations per elastic solve')
        axes[k, 1].set_ylabel('mean CG iterations per elastic solve')
        axes[k, 2].set_ylabel('CG iterations x 1000 (dual x 2)')
    axes[0, 0].legend(title='grey band: l/h < 2 (under-resolved)', loc='upper left', fontsize=11, title_fontsize=11)
    axes[0, 2].legend(loc='upper left')
    fig.suptitle(vtitle, fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', f'ASRpf_mesh_cg_vs_lh_{vkey}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('->', out)
