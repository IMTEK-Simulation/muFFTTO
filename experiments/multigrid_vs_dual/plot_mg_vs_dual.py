"""Multigrid vs dual preconditioner on AT2 fracture (polydisperse pores, l/h = 2): figures from the iterations CSVs.

    ./sync_nemo.sh && python plot_mg_vs_dual.py [sGc|fixed]

Series (argument): 'sGc' (default) = fixed strength, Gc = 0.016 l (l = 2h), so every grid cracks inside the load path;
'fixed' = Gc = 5e-4 on every grid (strength grows like 1/sqrt(l): 512^2/1024^2 do not crack by E_xx = 0.08).
The 64^2 runs belong to both series (Gc = 5e-4 = 0.016 / 32).

Methods: MG V(2,2) (library setup: 15 warm power iterations), MG V(1,1) (3 warm power iterations), dual;
each with the CG energy test d_min = 10 (solid, filled) and d_min = 0 (dashed, open).

figures/mg_vs_dual_summary_<series>.png  (a) mean elastic CG iterations per solve vs grid (laptop 64/128 + NEMO),
                                         (b) elastic GPU time per solve (NEMO L40S, 256-1024, incl. the MG setup),
                                         (c) whole-run elastic time / dual (same d_min), (d) CG per solve along the run
                                         at the largest grid with all methods
figures/mg_vs_dual_stress_<series>.png   stress-strain curves (all methods must coincide)
The table is printed.
"""
import glob
import os
import re
import sys

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({'font.size': 13, 'axes.labelsize': 13, 'legend.fontsize': 11})
METHODS = {'MG': ('#6a3d9a', 'o', 'MG V(2,2)'), 'MG11': ('#b07cd8', '^', 'MG V(1,1)'), 'dual': ('#d62728', 'D', 'dual')}
DELAYS = {'': ('-', 'full', r'$d_{\min}=10$'), '_d0': ('--', 'none', r'$d_{\min}=0$')}
SERIES = sys.argv[1] if len(sys.argv) > 1 else 'sGc'
SERIES_TAG = {'sGc': '_sGc', 'fixed': ''}[SERIES]
NAME = re.compile(r'AT2_N(\d+)_(MG|dual)_polydisperse_pores_l[\d.e-]+_lh2(_sGc)?(_v11(?:_pow3)?)?(_d0)?_iterations\.csv')


def load_runs():
    """{(N, method, delay): dict(data, source 'nemo'|'laptop', log)}"""
    runs = {}
    for source, folder in (('laptop', 'exp_data'), ('nemo', 'exp_data/nemo')):
        for f in glob.glob(os.path.join(here, folder, 'AT2_N*_iterations.csv')):
            m = NAME.fullmatch(os.path.basename(f))
            if not m:
                continue
            N, tag = int(m.group(1)), m.group(3) or ''
            if tag != SERIES_TAG and not (N == 64 and tag == ''):          # 64^2: Gc 5e-4 in both series
                continue
            method = 'MG11' if m.group(4) else m.group(2)
            delay = m.group(5) or ''
            data = np.atleast_1d(np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None))
            if data.size == 0:
                continue
            key = (N, method, delay)
            if key in runs and runs[key]['source'] == 'nemo':
                continue                                      # NEMO (GPU) preferred when both exist
            suffix = tag + (m.group(4) or '') + delay
            runs[key] = dict(data=data, source=source, log=find_log(N, m.group(2), suffix))
    return runs


def find_log(N, P, suffix):
    for name in (f'logs/nemo/AT2_N{N}_{P}_lh2{suffix}.log',
                 f'logs/AT2_N{N}_{P}_polydisperse_pores_lh2_lh2{suffix}.log',
                 f'logs/AT2_N{N}_{P}_polydisperse_pores_lh2{suffix}.log'):
        if os.path.exists(os.path.join(here, name)):
            return os.path.join(here, name)
    return None


def finished(run):
    return run['log'] is not None and 'Elapsed time' in open(run['log']).read()


def stress_strain(log):
    strain, stress = [0.0], [0.0]
    for line in open(log):
        m = re.match(r'step\s+\d+\s+E_xx = ([\d.]+)\s+sigma_xx = ([-\d.e+]+)', line)
        if m:
            strain.append(float(m.group(1)))
            stress.append(float(m.group(2)))
    return np.array(strain), np.array(stress)


def style(method, delay, **kw):
    color, marker, label = METHODS[method]
    ls, fill, dlabel = DELAYS[delay]
    return dict(color=color, ls=ls, marker=marker, fillstyle=fill, ms=8, lw=2, label=f'{label}, {dlabel}', **kw)


def main():
    runs = load_runs()
    grids = sorted({k[0] for k in runs})
    print(f'series {SERIES}')
    print(f'{"N":>5s} {"method":6s} {"d_min":5s} {"src":6s} {"done":4s} {"solves":>6s} {"CG/solve":>8s} {"max":>4s} '
          f'{"t_u [s]":>8s} {"ms/solve":>8s} {"setup":>6s} {"K/solve":>8s} {"G/solve":>8s} {"peak sigma":>10s}')
    for key in sorted(runs):
        r, d = runs[key], runs[key]['data']
        peak = stress_strain(r['log'])[1].max() if r['log'] else np.nan
        print(f'{key[0]:5d} {key[1]:6s} {"0" if key[2] else "10":5s} {r["source"]:6s} '
              f'{"yes" if finished(r) else "no":4s} {len(d):6d} {d["cg_u"].mean():8.1f} {d["cg_u"].max():4d} '
              f'{d["time_u"].sum():8.1f} {1e3 * d["time_u"].mean():8.1f} '
              f'{d["time_setup_u"].sum() / d["time_u"].sum():6.0%} '
              f'{(d["kwork_u"] + d["kwork_setup_u"]).mean():8.1f} {d["gwork_u"].mean():8.1f} {peak:10.4e}')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))
    ax_cg, ax_t, ax_ratio, ax_path = axes.ravel()
    for method in METHODS:
        for delay in DELAYS:
            keys = [(N, method, delay) for N in grids if (N, method, delay) in runs]
            if keys:
                ax_cg.plot([k[0] for k in keys], [runs[k]['data']['cg_u'].mean() for k in keys], **style(method, delay))
            gpu = [k for k in keys if runs[k]['source'] == 'nemo']
            if gpu:
                ax_t.plot([k[0] for k in gpu], [1e3 * runs[k]['data']['time_u'].mean() for k in gpu],
                          **style(method, delay))
            if method == 'dual':
                continue
            pairs = [N for N in grids if all(
                (N, m, delay) in runs and runs[(N, m, delay)]['source'] == 'nemo' and finished(runs[(N, m, delay)])
                for m in (method, 'dual'))]
            if pairs:
                ax_ratio.plot(pairs, [runs[(N, method, delay)]['data']['time_u'].sum()
                                      / runs[(N, 'dual', delay)]['data']['time_u'].sum() for N in pairs],
                              **style(method, delay))
    ax_ratio.axhline(1.0, color='#d62728', lw=1.5)
    ax_ratio.text(0.02, 0.97, 'above 1: dual faster\nbelow 1: multigrid faster', transform=ax_ratio.transAxes,
                  va='top', fontsize=12)
    complete = [N for N in grids if all((N, m, '') in runs for m in METHODS)]
    if complete:
        N = complete[-1]
        for method in METHODS:
            d = runs[(N, method, '')]['data']
            color, _, label = METHODS[method]
            ax_path.plot(np.arange(1, len(d) + 1), d['cg_u'], color=color, lw=1, label=label)
        ax_path.set_title(f'(d) {N}$^2$, $d_{{\\min}}=10$: every elastic solve', fontsize=13)
        ax_path.set_xlabel('elastic solve (all staggered iterations)')
        ax_path.set_ylabel('CG iterations')
        ax_path.set_ylim(bottom=0)
        ax_path.set_xlim(left=0)
        ax_path.legend()
    for ax in (ax_cg, ax_t, ax_ratio):
        ax.set_xscale('log', base=2)
        ax.set_xticks(grids)
        ax.set_xticklabels([str(N) for N in grids])
        ax.minorticks_off()
        ax.set_xlabel('grid N (N$^2$ pixels), $l/h = 2$')
        ax.set_ylim(bottom=0)
    ax_cg.set_ylabel('mean CG iterations per elastic solve')
    ax_cg.set_title('(a) iterations', fontsize=13)
    ax_t.set_ylabel('mean elastic time per solve [ms]')
    ax_t.set_title('(b) GPU time per solve (NEMO L40S, incl. MG setup)', fontsize=13)
    ax_ratio.set_ylabel('elastic time / dual (whole run)')
    ax_ratio.set_title('(c) GPU time relative to the dual', fontsize=13)
    ax_cg.legend(ncol=1)
    fig.suptitle('AT2 fracture, polydisperse pores (contrast $10^{-3}$), $k = 10^{-4}$, '
                 + ('$G_c = 0.016\\,l$ (fixed strength)' if SERIES == 'sGc' else '$G_c = 5\\cdot 10^{-4}$'),
                 fontsize=14)
    fig.tight_layout()
    out = os.path.join(here, 'figures', f'mg_vs_dual_summary_{SERIES}.png')
    fig.savefig(out, dpi=150)
    print('saved', out)

    fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 4.0), squeeze=False)
    for ax, N in zip(axes[0], grids):
        for method in METHODS:
            for delay in DELAYS:
                r = runs.get((N, method, delay))
                if r and r['log']:
                    e, s = stress_strain(r['log'])
                    st = style(method, delay)
                    ax.plot(e, s, color=st['color'], ls=st['ls'], lw=2.5 if method == 'dual' else 1.2,
                            label=st['label'])
        ax.set_title(f'{N}$^2$', fontsize=13)
        ax.set_xlabel(r'$\bar\varepsilon_{xx}$')
        ax.set_ylabel(r'$\bar\sigma_{xx}$')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(here, 'figures', f'mg_vs_dual_stress_{SERIES}.png')
    fig.savefig(out, dpi=150)
    print('saved', out)


if __name__ == '__main__':
    main()
