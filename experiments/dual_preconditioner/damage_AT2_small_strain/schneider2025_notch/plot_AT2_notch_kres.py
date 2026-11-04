"""
Single notch under tension (Schneider & Kaestner, FFEMS 2025, Sec. 4.1.1 / Fig. 14; see README.md), small strain, runs of
run_AT2_notch_kres.sh: residual stiffness k_res = 1e-6, 1e-5, 1e-4 (notch stiffness = k_res), Green / Green-Jacobi /
dual, 64^2 and 128^2.

    left    stress-strain curves (128^2, all k_res and preconditioners; stress in MPa: unit E = 210 GPa)
    middle  elastic CG iterations per solve over the load (128^2, k_res = 1e-6; raw iterations, dual 2 per iteration)
    right   total elastic work of the run (CG iterations x K applications, dual 2) against k_res, 64^2 and 128^2

CASE (env): notch (default, run_AT2_notch_kres.sh) or ch (the Cahn-Hilliard two-phase microstructure,
run_AT2_cahn_hilliard_kres.sh).

usage:  [CASE=ch] python plot_AT2_notch_kres.py     writes figures/AT2_{notch,cahn_hilliard}_kres.png, prints the table
"""
import os

import numpy as np
import netCDF4
import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.realpath(__file__))
PRECS = [('Green', 'C2', '--', 1, 'Green'), ('Green_Jacobi', 'C0', ':', 1, 'Green-Jacobi'), ('dual', 'C3', '-', 2, 'dual')]
KRES = ('1e-06', '1e-05', '0.0001')
E_MPA = 210e3
CASE = os.environ.get('CASE', 'notch')
TAG, NAME, TITLE = {'notch': ('_l0.02_k{k}_notch', 'notch', 'single notch'),
                    'ch': ('_incl0.1_l0.03125_k{k}_ch', 'cahn_hilliard', 'Cahn-Hilliard two-phase')}[CASE]


def load(n, p, k):
    base = os.path.join(HERE, 'exp_data', f'AT2_N{n}_{p}' + TAG.format(k=k))
    log = base.replace('exp_data', 'logs') + '.log'
    if not (os.path.exists(log) and 'Elapsed time' in open(log).read()):
        return None
    with netCDF4.Dataset(base + '.nc') as ds:
        strain, stress = np.asarray(ds['load'][:]), np.asarray(ds['macro_stress'][:, 1, 1]) * E_MPA
    it = np.genfromtxt(base + '_iterations.csv', delimiter=',', names=True, dtype=None, encoding=None)
    return strain, stress, it


fig, axes = plt.subplots(1, 3, figsize=(20, 6.3), layout='constrained', dpi=140)
print(f'{"N":>4s} {"k_res":>7s} | ' + ' '.join(f'{q[4]:>22s}' for q in PRECS) + '   (total work | elastic solves | peak MPa)')
for n in (64, 128):
    for k in KRES:
        row = []
        for p, c, ls, fac, label in PRECS:
            r = load(n, p, k)
            row.append('                     -' if r is None else
                       f'{fac * r[2]["cg_u"].sum():8.0f} {len(r[2]):5d} {r[1].max():6.1f}')
        print(f'{n:4d} {float(k):7.0e} | ' + ' '.join(row))
for k, alpha in zip(KRES, (1.0, 0.6, 0.35)):
    for p, c, ls, fac, label in PRECS:
        r = load(128, p, k)
        if r is not None:
            axes[0].plot(np.r_[0, r[0]] * 1e3, np.r_[0, r[1]], ls=ls, color=c, lw=2, alpha=alpha,
                         label=f'{label}, k$_{{res}}$ = {float(k):.0e}')
if CASE == 'notch':
    axes[0].axvline(0.82, color='0.5', lw=1, ls='-.')
    axes[0].text(0.83, 10, 'paper: crack\ninitiation 0.82e-3', fontsize=10, color='0.4')
axes[0].set_xlabel(r'$\bar\varepsilon_{yy}$ [10$^{-3}$]', fontsize=12), axes[0].set_ylabel(r'$\bar\sigma_{yy}$ [MPa]', fontsize=12)
axes[0].set_title(f'{TITLE}: stress-strain, 128$^2$', fontsize=13), axes[0].legend(fontsize=8.5, ncol=1)
for p, c, ls, fac, label in PRECS:
    r = load(128, p, '1e-06')
    if r is not None:
        it = r[2]
        st = np.unique(it['step'])
        axes[1].plot([it['load'][it['step'] == s][0] * 1e3 for s in st], [it['cg_u'][it['step'] == s].mean() for s in st],
                     ls=ls, color=c, lw=2, label=label)
axes[1].set_yscale('log')
axes[1].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
axes[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
axes[1].set_xlabel(r'$\bar\varepsilon_{yy}$ [10$^{-3}$]', fontsize=12)
axes[1].set_ylabel('CG iterations per elastic solve\n(mean over the load step)', fontsize=12)
axes[1].set_title('128$^2$, k$_{res}$ = 1e-6', fontsize=13), axes[1].legend(fontsize=11)
for n, mk, lsn in ((64, 'o', '--'), (128, 's', '-')):
    for p, c, ls, fac, label in PRECS:
        pts = [(float(k), fac * load(n, p, k)[2]['cg_u'].sum() / 1e3) for k in KRES if load(n, p, k) is not None]
        if pts:
            axes[2].plot(*np.array(pts).T, ls=lsn, marker=mk, ms=8, lw=2, color=c, label=f'{label}, {n}$^2$')
axes[2].set_xscale('log'), axes[2].set_xticks([float(k) for k in KRES])
axes[2].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.0e}'))
axes[2].xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axes[2].set_ylim(0, None)
axes[2].set_xlabel('residual stiffness k$_{res}$' + (' (= notch stiffness)' if CASE == 'notch' else ''), fontsize=12)
axes[2].set_ylabel('total elastic work of the run [thousands]\n(CG iterations, dual x 2)', fontsize=12)
axes[2].set_title(f'{TITLE}: cost against k$_{{res}}$', fontsize=13)
axes[2].legend(fontsize=9.5, ncol=2)
for ax in axes:
    ax.tick_params(labelsize=11), ax.grid(True, color='0.9')
out = os.path.join(HERE, 'figures', f'AT2_{NAME}_kres.png')
fig.savefig(out)
print(f'wrote {out}')
