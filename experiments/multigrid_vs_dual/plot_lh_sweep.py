"""l/h sweep of the multigrid_vs_dual study: figures and table from the iterations CSVs.

    ./sync_nemo.sh && python plot_lh_sweep.py

Runs: AT2 polydisperse pores, l = (l/h) h, Gc = 0.016 l (fixed strength), l/h = 0.5 ... 4; grids 64, 128 (laptop CPU,
run_lh_sweep_laptop.sh) and 256, 512, 1024 (NEMO L40S, sbatch_lh_sweep.sh, the six configurations of one (N, l/h)
sequentially in one job). Configurations: dual, MG V(1,1), MG V(2,2) x d_min 10 / 0.
Main data: runs tagged _lb (MG smoother interval from the rigorous bound lambda_max(D^-1 A) <= 3.523, no power
iteration). First pass (no tag, MG_POWER=3 warm power estimate): kept as config + '_power' for the robustness figure
(the estimate missed the crack mode at the 1024^2 cut-through -> single solves with up to 12000 CG iterations).
The adaptive-power test runs (_pa) are ignored.

figures/lh_sweep_iterations.png   mean elastic CG iterations per solve vs l/h, one panel per method, a line per grid
figures/lh_sweep_time_ratio.png   elastic wall time of the whole run, MG / dual (same d_min), per (grid, l/h);
                                  < 1: multigrid faster. 64/128: laptop CPU, 256-1024: L40S GPU
figures/lh_sweep_robustness.png   largest CG count of a single elastic solve: power estimate vs bound, and the dual
figures/lh_sweep_damage.png       final damage (dual, d_min 10) for every l/h at the largest grid with all of them
exp_data/lh_sweep_summary.csv     one row per run
"""
import glob
import os
import re

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

here = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 11})
NAME = re.compile(r'AT2_N(\d+)_(MG|dual)_polydisperse_pores_l[\d.e-]+_sw_lh([\d.]+)(_v11|_v22)?(_d0)?(_pa)?(_lb)?'
                  r'_iterations\.csv')
CONFIG_LABEL = {'dual': 'dual', 'MG11': 'MG V(1,1)', 'MG22': 'MG V(2,2)'}
GRID_STYLE = {64: ('o', ':'), 128: ('s', ':'), 256: ('^', '-'), 512: ('D', '-'), 1024: ('v', '-')}
SHADES = {'dual': {64: '#f4a3a3', 128: '#ea6b6b', 256: '#d62728', 512: '#a51d1d', 1024: '#6b0f0f'},
          'MG11': {64: '#e3cdf5', 128: '#c9a3eb', 256: '#a774d6', 512: '#7e48b5', 1024: '#4e2477'},
          'MG22': {64: '#e3cdf5', 128: '#c9a3eb', 256: '#a774d6', 512: '#7e48b5', 1024: '#4e2477'}}


def config_of(method, tag, delay):
    return ('dual' if method == 'dual' else {'_v11': 'MG11', '_v22': 'MG22'}[tag]) + (delay or '')


def find_log(N, config, lh, pa=''):
    for name in (f'logs/nemo/AT2_N{N}_{config}_sw_lh{lh}{pa}.log', f'logs/AT2_N{N}_{config}_sw_lh{lh}{pa}.log'):
        path = os.path.join(here, name)
        if os.path.exists(path):
            return path
    return None


def load_runs():
    """{(N, lh, config): run} from the _lb pass; the first pass (power estimate) as (N, lh, config + '_power')"""
    runs = {}
    for source, folder in (('laptop', 'exp_data'), ('nemo', 'exp_data/nemo')):
        for f in glob.glob(os.path.join(here, folder, 'AT2_N*_sw_lh*_iterations.csv')):
            m = NAME.fullmatch(os.path.basename(f))
            if not m or m.group(6):
                continue                                       # _smoke runs, adaptive-power test runs (_pa)
            N, lh = int(m.group(1)), float(m.group(3))
            config = config_of(m.group(2), m.group(4), m.group(5))
            lb = m.group(7) or ''
            data = np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None))
            if data.size == 0:
                continue
            log = find_log(N, config, m.group(3), lb)
            text = open(log).read() if log else ''
            stress = [float(v) for v in re.findall(r'sigma_xx = ([-\d.e+]+)', text)]
            run = dict(data=data, source=source, log=log, done='Elapsed time' in text,
                       peak=max(stress) if stress else np.nan, steps=len(stress),
                       cut='cut through' in text, damage=f.replace('_iterations.csv', '_final_damage.npy'))
            runs[(N, lh, config if lb else config + '_power')] = run
    return runs


def table(runs):
    rows = []
    for (N, lh, config), r in sorted(runs.items()):
        d = r['data']
        rows.append(dict(N=N, lh=lh, config=config, source=r['source'], done=int(r['done']), cut=int(r['cut']),
                         steps=r['steps'], solves=len(d), cg_mean=d['cg_u'].mean(), cg_max=int(d['cg_u'].max()),
                         time_u=d['time_u'].sum(), time_d=d['time_d'].sum(),
                         setup_share=d['time_setup_u'].sum() / max(d['time_u'].sum(), 1e-30),
                         k_per_solve=(d['kwork_u'] + d['kwork_setup_u']).mean(), g_per_solve=d['gwork_u'].mean(),
                         peak_sigma=r['peak']))
    out = os.path.join(here, 'exp_data', 'lh_sweep_summary.csv')
    with open(out, 'w') as f:
        f.write(','.join(rows[0]) + '\n')
        for row in rows:
            f.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in row.values()) + '\n')
    print(f'{"N":>5s} {"l/h":>4s} {"config":8s} {"src":6s} {"done":4s} {"cut":3s} {"solves":>6s} {"CG/solve":>8s} '
          f'{"max":>4s} {"t_u [s]":>8s} {"setup":>5s} {"K/solve":>7s} {"G/solve":>7s} {"peak":>9s}')
    for row in rows:
        print(f'{row["N"]:5d} {row["lh"]:4g} {row["config"]:8s} {row["source"]:6s} {"yes" if row["done"] else "no":4s} '
              f'{"yes" if row["cut"] else "no":3s} {row["solves"]:6d} {row["cg_mean"]:8.1f} {row["cg_max"]:4d} '
              f'{row["time_u"]:8.1f} {row["setup_share"]:5.0%} {row["k_per_solve"]:7.1f} {row["g_per_solve"]:7.1f} '
              f'{row["peak_sigma"]:9.3e}')
    print('saved', out)


def plot_iterations(runs, grids, lhs):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), sharey='row')
    for row, delay in enumerate(('', '_d0')):
        for ax, method in zip(axes[row], ('dual', 'MG11', 'MG22')):
            config = method + delay
            for N in grids:
                keys = [(N, lh, config) for lh in lhs if (N, lh, config) in runs and runs[(N, lh, config)]['done']]
                if keys:
                    marker, ls = GRID_STYLE[N]
                    ax.plot([k[1] for k in keys], [runs[k]['data']['cg_u'].mean() for k in keys],
                            color=SHADES[method][N], marker=marker, ls=ls, lw=2, ms=7,
                            label=f'{N}$^2$' + (' (CPU)' if N <= 128 else ''))
            ax.set_title(f'{CONFIG_LABEL[method]}, $d_{{\\min}} = {0 if delay else 10}$', fontsize=13)
            ax.set_xlabel('$l / h$')
            ax.set_xticks(lhs)
            ax.set_xticklabels([f'{v:g}' for v in lhs])
            ax.set_xlim(0, max(lhs) + 0.3)
            ax.set_ylim(bottom=0)
            ax.set_ylabel('mean CG iterations per elastic solve')
            ax.tick_params(labelleft=True)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=10)
    fig.suptitle('AT2, polydisperse pores, $G_c = 0.016\\,l$ (fixed strength): elastic CG iterations', fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', 'lh_sweep_iterations.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


def plot_time_ratio(runs, grids, lhs):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    norm = TwoSlopeNorm(vmin=0.25, vcenter=1.0, vmax=4.0)
    for row, delay in enumerate(('', '_d0')):
        for ax, method in zip(axes[row], ('MG11', 'MG22')):
            ratio = np.full((len(lhs), len(grids)), np.nan)
            for i, lh in enumerate(lhs):
                for j, N in enumerate(grids):
                    a, b = runs.get((N, lh, method + delay)), runs.get((N, lh, 'dual' + delay))
                    if a and b and a['done'] and b['done']:
                        ratio[i, j] = a['data']['time_u'].sum() / b['data']['time_u'].sum()
            image = ax.imshow(ratio, cmap='PuOr', norm=norm, origin='lower', aspect='auto')
            for i in range(len(lhs)):
                for j in range(len(grids)):
                    if np.isfinite(ratio[i, j]):
                        ax.text(j, i, f'{ratio[i, j]:.2f}', ha='center', va='center', fontsize=12,
                                color='white' if abs(np.log(ratio[i, j])) > 0.8 else 'black')
            ax.set_xticks(range(len(grids)))
            ax.set_xticklabels([f'{N}' + ('\n(CPU)' if N <= 128 else '\n(GPU)') for N in grids])
            ax.set_yticks(range(len(lhs)))
            ax.set_yticklabels([f'{v:g}' for v in lhs])
            ax.set_xlabel('grid N')
            ax.set_ylabel('$l / h$')
            ax.set_title(f'{CONFIG_LABEL[method]} / dual, $d_{{\\min}} = {0 if delay else 10}$', fontsize=13)
            fig.colorbar(image, ax=ax, label='elastic time MG / dual (< 1: MG faster)')
    fig.suptitle('Whole-run elastic wall time (incl. MG setup), multigrid relative to the dual', fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', 'lh_sweep_time_ratio.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


def plot_robustness(runs, grids, lhs):
    """largest CG count of one elastic solve per run (d_min 10): MG with the power estimate vs with the bound"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (method, label) in zip(axes, (('MG11', 'MG V(1,1)'), ('MG22', 'MG V(2,2)'), ('dual', 'dual'))):
        for variant, ls, fill, vlabel in (('', '-', 'full', 'bound' if method != 'dual' else ''),
                                          ('_power', '--', 'none', 'power estimate')):
            if method == 'dual' and variant:
                continue
            for N in grids:
                keys = [(N, lh, method + variant) for lh in lhs if (N, lh, method + variant) in runs
                        and runs[(N, lh, method + variant)]['done']]
                if keys:
                    marker, _ = GRID_STYLE[N]
                    ax.plot([k[1] for k in keys], [runs[k]['data']['cg_u'].max() for k in keys],
                            color=SHADES[method][N], marker=marker, ls=ls, fillstyle=fill, lw=2, ms=7,
                            label=f'{N}$^2$' + (f', {vlabel}' if vlabel else ''))
        ax.set_yscale('log')
        ax.set_yticks([10, 30, 100, 300, 1000, 3000, 10000])
        ax.set_yticklabels(['10', '30', '100', '300', '1000', '3000', '10000'])
        ax.minorticks_off()
        ax.set_xticks(lhs)
        ax.set_xticklabels([f'{v:g}' for v in lhs])
        ax.set_xlabel('$l / h$')
        ax.set_ylabel('largest CG count of one elastic solve')
        ax.tick_params(labelleft=True)
        ax.set_title(f'{label}, $d_{{\\min}} = 10$', fontsize=13)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle('Robustness: MG smoother bound from a warm power estimate (dashed) vs the rigorous bound (solid)',
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', 'lh_sweep_robustness.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


def plot_damage(runs, grids, lhs):
    candidates = [N for N in grids if all((N, lh, 'dual') in runs and os.path.exists(runs[(N, lh, 'dual')]['damage'])
                                          for lh in lhs)]
    if not candidates:
        candidates = [N for N in grids if any((N, lh, 'dual') in runs for lh in lhs)]
    if not candidates:
        return
    N = candidates[-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.5))
    for ax, lh in zip(axes.ravel(), lhs):
        r = runs.get((N, lh, 'dual'))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'$l/h = {lh:g}$', fontsize=13)
        if r is None or not os.path.exists(r['damage']):
            continue
        d = np.load(r['damage'])
        ax.imshow(d.T, origin='lower', cmap='inferno', vmin=0, vmax=1, interpolation='nearest',
                  extent=(0, 1, 0, 1))
        ax.set_title(f'$l/h = {lh:g}$, ' + ('cut through' if r['cut'] else 'end of load path'), fontsize=13)
    fig.suptitle(f'final damage, {N}$^2$ (dual run)', fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', 'lh_sweep_damage.png')
    fig.savefig(out, dpi=max(150, int(N / 4.5)))
    print('saved', out)


def main():
    runs = load_runs()
    if not runs:
        print('no sweep runs yet')
        return
    grids = sorted({k[0] for k in runs})
    lhs = sorted({k[1] for k in runs})
    table(runs)
    plot_robustness(runs, grids, lhs)
    plot_iterations(runs, grids, lhs)
    plot_time_ratio(runs, grids, lhs)
    plot_damage(runs, grids, lhs)


if __name__ == '__main__':
    main()
