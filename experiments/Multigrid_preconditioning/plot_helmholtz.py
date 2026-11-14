"""
Helmholtz-preconditioned L-BFGS for TO.
Figure 1 (laptop, Green-Jacobi mechanics, CH start, eta = 4h): dependence on the smoothing strength c of
M = (I + c eta^2 (-Lap_h))^-1 (smoothing length l = sqrt(c) eta), c = 0 = plain L-BFGS: objective evaluations to the
stop and final objective, 32^2 and 64^2.
Figure 2 (NEMO, multigrid mechanics, c = 1): objective vs evaluations, plain NuMPI L-BFGS vs Helmholtz, per grid,
eta = 2h and 4h.

    ./sync_nemo_TO.sh && python plot_helmholtz.py
"""
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent


def evals(path):
    if not path.exists():
        return None
    ev = re.findall(r'^EVAL (\d+) f ([\d.eE+-]+) wall ([\d.]+)', path.read_text(), flags=re.M)
    return np.array([[float(a), float(b), float(c)] for a, b, c in ev]) if ev else None


plt.rcParams.update({'font.size': 13})

# ---------------------------------------------------------------- figure 1: c-dependence
CS = [0, 0.015625, 0.0625, 0.25, 1, 4, 16, 64]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.4))
print('c-dependence (laptop): evaluations / final f')
for N, color, mk in [(32, '#c2a5cf', 'o'), (64, '#762a83', 's')]:
    xs, ne, ff = [], [], []
    for i, c in enumerate(CS):
        name = f'dopt_N{N}_lbfgs.log' if c == 0 else (f'dopt_N{N}_lbfgs_helm.log' if c == 1 else
                                                      f'dopt_N{N}_lbfgs_helm_c{c:g}.log')
        d = evals(HERE / 'logs' / name)
        if d is None:
            continue
        xs.append(i); ne.append(d[-1, 0]); ff.append(d[-1, 1])
        print(f'  N {N:3d}  c {c:<9g} {int(d[-1, 0]):5d} evals  f {d[-1, 1]:.5f}')
    ax1.plot(xs, ne, color=color, marker=mk, lw=2, ms=8, label=f'{N}²')
    ax2.plot(xs, np.array(ff) / min(ff), color=color, marker=mk, lw=2, ms=8, label=f'{N}²')
labels = ['plain'] + ['1/64', '1/16', '1/4', '1', '4', '16', '64']
for ax in (ax1, ax2):
    ax.set_xticks(range(len(CS)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('smoothing strength c   (length ℓ = √c · η)')
    ax.grid(alpha=0.3)
    ax.legend()
ax1.set_ylabel('objective evaluations to the stop')
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
ax1.set_title('cost (each evaluation = 6 linear solves)')
ax2.set_ylabel('final objective / best')
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.3f}'))
ax2.set_title('design quality')
fig.suptitle('L-BFGS with Helmholtz H₀ = (I + c η² (−Δ))⁻¹, η = 4h, Cahn-Hilliard start', y=1.0)
fig.tight_layout()
out1 = HERE / 'figures' / 'helmholtz_c_dependence.png'
fig.savefig(out1, dpi=150, bbox_inches='tight')
print(out1)

# ---------------------------------------------------------------- figure 2: NEMO runs
L = HERE / 'logs' / 'nemo_TO'
GRIDS = [64, 128, 256, 512]
SHADES = {64: 0.35, 128: 0.55, 256: 0.75, 512: 1.0}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
print('NEMO (multigrid mechanics): evaluations / f, plain vs Helmholtz')
for ax, E in zip(axes, [2, 4]):
    for N in GRIDS:
        for tag, ls, base in [('', '--', np.array([0.55, 0.45, 0.65])), ('_lbfgs_helm', '-', np.array([0.46, 0.16, 0.51]))]:
            d = evals(L / f'TO_N{N}_MG_bi_s0_eta{E}h{tag}.log')
            if d is None:
                continue
            col = 1 - SHADES[N] * (1 - base)
            lab = f'{N}² ' + ('Helmholtz' if tag else 'plain')
            ax.plot(d[:, 0], np.minimum.accumulate(d[:, 1]), color=col, ls=ls, lw=2, label=lab)
            print(f'  eta {E}h N {N:3d} {"helm " if tag else "plain"} {int(d[-1, 0]):5d} evals  f {d[-1, 1]:.5f}')
    ax.set_xlabel('objective evaluations')
    ax.set_ylabel('objective (best so far)')
    ax.set_title(f'η = {E}h: plain L-BFGS (dashed) vs Helmholtz c = 1 (solid)')
    ax.set_xlim(left=0)
    ax.set_ylim(0.6, 2.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9.5, ncol=2, loc='upper right')
fig.tight_layout()
out2 = HERE / 'figures' / 'helmholtz_nemo.png'
fig.savefig(out2, dpi=150)
print(out2)
