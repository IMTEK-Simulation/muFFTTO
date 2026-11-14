"""2D eta-continuation test (run_TO_eta_continuation.sh) against the direct 128^2 runs of run_TO_CH_starts.sh.
All Green-Jacobi, contrast 1e4, w = 0.2, a = 1, final grid 128^2 with eta = 1/32 (4h). Per run: final design (exact
pixels), objective, L-BFGS iterations and CG iterations (mechanical + adjoint solves) of the final stage and of all
stages, periodic connected components of solid / void at rho = 0.5.
    python plot_TO_eta_continuation.py  ->  figures/TO_eta_continuation.png
"""
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'data', 'example_2D_elasticity_TO')
L = os.path.join(HERE, 'logs')
F = 'N{}_soft0.0001_energy-warm_Green_Jacobi_eta_{}_pfw0.2_l_bfgs{}_final.nc'
# label, stages [(grid, eta, suffix, log)], final stage last
RUNS = [('direct, sine start', [(128, 0.03125, '_start_sine', 'TO_CH_starts/sine')]),
        ('direct, CH start bi\\_s0', [(128, 0.03125, '_start_bi_s0', 'TO_CH_starts/bi_s0')]),
        ('U: sine, $64^2$ $8h$ $\\to$ $128^2$ $4h$', [(64, 0.125, '_cont_s1', 'TO_eta_continuation/s1_N64_eta8h'),
                                                     (128, 0.03125, '_cont_U', 'TO_eta_continuation/U_N128_Green_Jacobi')]),
        ('S: sine, $64^2$ $8h \\to 4h$ $\\to$ $128^2$ $4h$', [(64, 0.125, '_cont_s1', 'TO_eta_continuation/s1_N64_eta8h'),
                                                          (64, 0.0625, '_cont_s2', 'TO_eta_continuation/s2_N64_eta4h'),
                                                          (128, 0.03125, '_cont_S', 'TO_eta_continuation/S_N128_Green_Jacobi')]),
        ('C: CH bi\\_s0, $64^2$ $8h$ $\\to$ $128^2$ $4h$', [(64, 0.125, '_cont_C1', 'TO_eta_continuation/C1_N64_eta8h'),
                                                          (128, 0.03125, '_cont_C', 'TO_eta_continuation/C_N128_Green_Jacobi')])]


def components(mask):
    lab, n = ndimage.label(mask)
    par = list(range(n + 1))

    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for ax in range(2):
        a, b = np.take(lab, 0, axis=ax), np.take(lab, -1, axis=ax)
        m = (a > 0) & (b > 0)
        for x, y in zip(a[m], b[m]):
            par[f(x)] = f(y)
    return len({f(k) for k in range(1, n + 1)})


def log_cost(name):
    txt = open(os.path.join(L, name + '.log')).read()
    cg = sum(int(x) for x in re.findall(r'CG (?:mech|adjoint)\s*=\s*(\d+)', txt))
    its = [int(m) for m in re.findall(r'^\s*(\d+)\s+[0-9.]+e[+-]\d+\s+[0-9.]+e[+-]\d+', txt, re.M)]
    return (max(its) if its else 0), cg


plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 13})
fig, axs = plt.subplots(1, len(RUNS), figsize=(4.0 * len(RUNS), 5.6), layout='constrained')
for ax, (label, stages) in zip(axs, RUNS):
    N, eta, suf, _ = stages[-1]
    with netCDF4.Dataset(os.path.join(D, F.format(N, eta, suf))) as d:
        rho = np.asarray(d['phase_field_solution'][0, 0])
    hist = os.path.join(D, F.format(N, eta, suf).replace('_final.nc', '_history.nc'))
    with netCDF4.Dataset(hist) as d:
        f_end = float(d['objective'][-1])
    costs = [log_cost(s[3]) for s in stages]
    it_last, cg_last = costs[-1]
    cg_all = sum(c for _, c in costs)
    it_all = sum(i for i, _ in costs)
    nv, ns = components(rho < 0.5), components(rho >= 0.5)
    ax.imshow(rho.T, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title(label, fontsize=13)
    ax.text(0.5, -0.03, f'$f = {f_end:.4f}$\n'
            f'final stage: {it_last} it, {cg_last / 1e3:.0f}k CG\n'
            f'all stages: {it_all} it, {cg_all / 1e3:.0f}k CG\n'
            f'void / solid pieces: {nv} / {ns}', transform=ax.transAxes, ha='center', va='top', fontsize=12)
    print(f'{label:55s} f={f_end:.4f}  final {it_last} it {cg_last} CG  all {it_all} it {cg_all} CG  void {nv} solid {ns}')
fig.suptitle(r'2D $\eta$-continuation, Green-Jacobi, contrast $10^4$: final designs on $128^2$, $\eta = 1/32$ (4h)',
             fontsize=15)
out = os.path.join(HERE, 'figures', 'TO_eta_continuation.png')
fig.savefig(out, dpi=130)
print('wrote', out)
