"""
Hourglass-control strength study (run_hg_test.sh): one-point hexahedra + muFFTTO hourglass control
(trilinear_hexahedron_1Q_hg, eps = fraction of the fully integrated hourglass stiffness) vs the 8-point hexahedron.
MG-preconditioned CG, 3D Cahn-Hilliard geometry ch_s12, contrast 1e4, 32^3 and 64^3, eta/h = 0 (sharp), 1, 2.
-> figures/hourglass_eps.png
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
D = HERE / 'exp_data' / 'scaling3d_1Q' / 'csv'
EPS = ['0.03', '0.1', '0.3', '1']
COL = {0.0: '#7b3294', 1.0: '#e66101', 2.0: '#1a9641'}
LS = {32: ('--', 'o', 'none'), 64: ('-', 's', None)}       # line style, marker, fill (none = open)
plt.rcParams.update({'font.size': 13})


def rows(name):
    return {float(r['eta_over_h']): r for r in csv.DictReader(open(D / name)) if r['method'] == 'mg'}


fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
handles = []
for N in (32, 64):
    ref = rows(f'mg3d_scaling_ch_s12_N{N}.csv')
    data = {e: rows(f'mg3d_scaling_ch_s12_N{N}_1Qhg{e}.csv') for e in EPS}
    x = [float(e) for e in EPS]
    for eta in (0.0, 1.0, 2.0):
        ls, mk, fill = LS[N]
        kw = dict(color=COL[eta], ls=ls, marker=mk, ms=8, lw=2, mfc=fill if fill else COL[eta])
        it = [int(data[e][eta]['iterations']) for e in EPS]
        err = [100 * (float(data[e][eta]['C11']) / float(ref[eta]['C11']) - 1) for e in EPS]
        h, = axes[0].plot(x, it, label=f'{N}³, η = {eta:g}h' if eta else f'{N}³, sharp', **kw)
        handles.append(h)
        axes[0].axhline(int(ref[eta]['iterations']), color=COL[eta], ls=ls, lw=1, alpha=0.4)
        axes[2 if eta == 0 else 1].plot(x, err, **kw)
for ax, title, ylab in zip(axes, ['MG-CG iterations (thin: 8-point)', 'C11 error vs 8-point, η ≥ h',
                                  'C11 error vs 8-point, sharp interface'],
                           ['iterations', 'C11 error [%]', 'C11 error [%]']):
    ax.set_xscale('log')
    ax.set_xticks([float(e) for e in EPS])
    ax.set_xticklabels(EPS)
    ax.minorticks_off()
    ax.set_xlabel('hourglass control ε (fraction of 8-point hourglass stiffness)')
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.3)
axes[0].set_ylim(0, 30)
axes[1].axhline(0, color='k', lw=0.8)
axes[2].set_ylim(-18, 0)
for ax in axes:
    ax.axvline(0.3, color='0.5', ls=':', lw=1.5)
axes[0].text(0.31, 27, 'default 0.3', color='0.4', fontsize=12)
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=12)
fig.suptitle('One-point hexahedra + hourglass control (muFFTTO trilinear_hexahedron_1Q_hg), MG preconditioner, ch_s12, '
             'contrast 1e4', fontsize=13)
fig.tight_layout(rect=(0, 0.08, 1, 0.97))
out = HERE / 'figures' / 'hourglass_eps.png'
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150)
print(out)
