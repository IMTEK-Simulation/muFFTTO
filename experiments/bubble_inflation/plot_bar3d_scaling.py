"""Mesh scaling of the 3D runs (pressurized_bar_3D.py), from the run logs: sealed bar (p 0 -> 12) and the pulse in the
periodic channel (length 3), corrected Green with the wall FFT (G_wall) vs projected Green (G).
Logs: runs/bar3d_n<N>_*/run.log (laptop) and runs/nemo/logs/bar3d_n<N>_*/run.log (NEMO); only complete runs.
Usage: python plot_bar3d_scaling.py   -> bar3d_scaling.png"""
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({'font.size': 13})
HERE = os.path.dirname(os.path.realpath(__file__))
MESHES = (32, 64, 128)
SHADE = {32: '#74c476', 64: '#31a354', 128: '#00441b'}
STY = {'G_wall': dict(ls='-', marker='^', label='corrected Green (wall FFT)', filled=True),
       'G': dict(ls='--', marker='s', label='projected Green', filled=False)}
CASES = {'sealed': ('_p12_{P}_gpu_', 12), 'pulse': ('_p12_per_L3_pulse_{P}_gpu_', 60)}


def parse(log):
    inc = []
    for line in open(log):
        m = re.match(r'inc +(\d+) +p +([\d.]+) .*newton +(\d+) +hessp +(\d+)', line)
        if m:
            t = re.search(r'([\d.]+) s(  GPU|$)', line.rstrip())
            inner = re.search(r'inner CG \[([\d, ]*)\]', line)
            inc.append(dict(p=float(m.group(2)), nit=int(m.group(3)), hessp=int(m.group(4)),
                            time=float(t.group(1)) if t else np.nan,
                            inner=[int(v) for v in inner.group(1).split(',')] if inner and inner.group(1) else []))
    done = any(line.startswith('elapsed') for line in open(log))
    return inc, done


def find(case, n, P):
    pattern, nb = CASES[case]
    logs = (glob.glob(os.path.join(HERE, 'runs', f'bar3d_n{n}_R0.3125' + pattern.format(P=P) + '*', 'run.log'))
            + glob.glob(os.path.join(HERE, 'runs', 'nemo', 'logs', f'bar3d_n{n}_R0.3125' + pattern.format(P=P) + '*',
                                     'run.log')))
    logs = [l for l in logs if '_At_' not in l]
    good = []
    for l in sorted(logs, key=os.path.getmtime):
        inc, done = parse(l)
        if done and len(inc) == nb:
            good.append((l, inc))
    return good[-1] if good else None


fig, axes = plt.subplots(2, 3, figsize=(20, 11))
summary = []
for row, case in enumerate(('sealed', 'pulse')):
    ax_t, ax_m, ax_r = axes[row]
    means = {P: {} for P in STY}
    for n in MESHES:
        for P, st in STY.items():
            r = find(case, n, P)
            if r is None:
                continue
            log, inc = r
            x = np.array([i['p'] for i in inc]) if case == 'sealed' else np.arange(1, len(inc) + 1)
            per = np.array([i['hessp'] / i['nit'] for i in inc])
            ax_t.plot(x, per, color=SHADE[n], ls=st['ls'], marker=st['marker'], ms=4 if case == 'pulse' else 6,
                      mfc=SHADE[n] if st['filled'] else 'white', lw=1.6, label=f'{st["label"]}, h = 1/{n}')
            tot_h, tot_n = sum(i['hessp'] for i in inc), sum(i['nit'] for i in inc)
            means[P][n] = tot_h / tot_n
            where = 'NEMO L40S' if '/nemo/' in log else 'laptop GPU'
            summary.append(f'{case:6s} 1/{n:<4d} {P:6s} {tot_h / tot_n:6.1f} per solve  {tot_n:4d} solves  '
                           f'{tot_h:6d} products  {sum(i["time"] for i in inc) / 60:6.1f} min ({where})')
    for P, st in STY.items():
        ns = sorted(means[P])
        ax_m.plot(ns, [means[P][n] for n in ns], color='#238b45' if P == 'G_wall' else '#74c476', ls=st['ls'],
                  marker=st['marker'], ms=9, mfc='#238b45' if st['filled'] else 'white', lw=2, label=st['label'])
        for n in ns:
            ax_m.annotate(f'{means[P][n]:.1f}', (n, means[P][n]), textcoords='offset points', xytext=(-12, 9),
                          fontsize=12, color='#238b45' if P == 'G_wall' else '#4a9e5c')
    ns = sorted(set(means['G']) & set(means['G_wall']))
    ax_r.plot(ns, [means['G'][n] / means['G_wall'][n] for n in ns], 'o-', color='#238b45', ms=9, lw=2)
    for n in ns:
        ax_r.annotate(f'{means["G"][n] / means["G_wall"][n]:.2f}', (n, means['G'][n] / means['G_wall'][n]),
                      textcoords='offset points', xytext=(-12, 9), fontsize=12)
    title = 'sealed bar, p = 0 -> 12' if case == 'sealed' else 'pulse in the periodic channel (length 3)'
    ax_t.set_xlabel('pressure p   (E = 100)' if case == 'sealed' else 'load step (1-12: amplitude ramp, 13-60: pulse travels)')
    ax_t.set_ylabel('Hessian products per Newton solve')
    ax_t.set_title(title)
    ax_t.set_xlim(0, None)
    ax_t.set_ylim(0, 1.5 * ax_t.get_ylim()[1])                       # room for the legend above the curves
    ax_t.grid(alpha=0.3)
    ax_t.legend(fontsize=10, ncol=2, loc='upper left', framealpha=0.95)
    for ax in (ax_m, ax_r):
        ax.set_xticks(MESHES, [str(n) for n in MESHES])
        ax.set_xlim(0, 140)
        ax.set_xlabel('voxels per unit length (1/h)')
        ax.grid(alpha=0.3)
    ax_m.set_ylabel('mean Hessian products per Newton solve')
    ax_m.set_ylim(0, 1.25 * max(max(v.values()) for v in means.values()))
    ax_m.set_title(title + ': mean per solve')
    ax_m.legend(loc='lower right')
    ax_r.set_ylabel('projected / corrected (per solve)')
    ax_r.set_ylim(0, 3.2)
    ax_r.set_title('advantage of the corrected Green')
fig.suptitle('3D hollow bar / channel, x faces clamped, Green preconditioner (C_ref = solid): mesh scaling')
fig.tight_layout()
out = os.path.join(HERE, 'bar3d_scaling.png')
fig.savefig(out, dpi=140, bbox_inches='tight')
print(out)
print('\n'.join(summary))
