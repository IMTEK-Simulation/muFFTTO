"""
Settings prototype (run_proto.sh): 2D TO, CH start bi_s11, MG mechanics + Helmholtz L-BFGS,
eta in {2h, 4h} x c in {1, 4, 16} x V(2,2) / V(1,1). Table + figure (final designs, exact pixels, dark = stiff,
labelled with evaluations, wall time, objective; wall-time bars).

    python plot_proto.py [--grid 128]
"""
import argparse
import re
from pathlib import Path

import numpy as np
import netCDF4
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=128)
parser.add_argument('--soft-exp', type=int, default=3, help='contrast 10^soft_exp (soft phase 10^-soft_exp)')
args = parser.parse_args()
N = args.grid
ETAS, CS, DS = [2, 4], [1, 4, 16], [2, 1]
DATA = HERE / 'data' / 'example_2D_elasticity_TO'


def load(E, C, Dg):
    log = HERE / 'logs' / 'proto' / f'soft{args.soft_exp}' / f'N{N}_eta{E}h_c{C}_V{Dg}.log'
    if not log.exists():
        return None
    t = log.read_text()
    ev = re.findall(r'EVAL (\d+) f (\S+) wall ([\d.]+)', t)
    its = [int(v) for v in re.findall(r'steps CG (?:mech|adjoint)\s*=\s*(\d+)', t)]
    if not ev:
        return None
    nc = DATA / f'N{N}_soft{10.0 ** -args.soft_exp:g}_energy-warm_MG_eta_{E / N}_pfw0.2_l_bfgs_proto_eta{E}h_c{C}_V{Dg}_final.nc'
    rho = np.asarray(netCDF4.Dataset(nc)['phase_field_solution'][:]).reshape(N, N) if nc.exists() else None
    return dict(evals=int(ev[-1][0]), f=float(ev[-1][1]), wall=float(ev[-1][2]), cg=np.mean(its) if its else np.nan,
                done='TOTAL wall' in t, rho=rho)


res = {(E, C, Dg): load(E, C, Dg) for E in ETAS for C in CS for Dg in DS}
print(f'{"eta":>4s} {"c":>3s} {"cycle":>7s} {"evals":>6s} {"wall s":>8s} {"CG/solve":>9s} {"f":>9s} {"grey":>6s}  status')
for (E, C, Dg), r in res.items():
    if r is None:
        print(f'{E:3d}h {C:3d}  V({Dg},{Dg})   (not started)')
        continue
    grey = ((r['rho'] > 0.1) & (r['rho'] < 0.9)).mean() if r['rho'] is not None else np.nan
    print(f'{E:3d}h {C:3d}  V({Dg},{Dg}) {r["evals"]:6d} {r["wall"]:8.0f} {r["cg"]:9.1f} {r["f"]:9.5f} {grey:6.3f}  '
          f'{"done" if r["done"] else "running"}')

plt.rcParams.update({'font.size': 11})
ncol = len(CS) * len(DS)
fig = plt.figure(figsize=(2.6 * ncol, 2.9 * len(ETAS) + 3.2))
gs = fig.add_gridspec(len(ETAS) + 1, ncol, height_ratios=[1] * len(ETAS) + [1.1])
for r_i, E in enumerate(ETAS):
    for c_i, (C, Dg) in enumerate([(C, Dg) for C in CS for Dg in DS]):
        ax = fig.add_subplot(gs[r_i, c_i])
        ax.set_xticks([]); ax.set_yticks([])
        r = res[E, C, Dg]
        if r is None or r['rho'] is None:
            ax.set_title(f'η={E}h c={C} V({Dg},{Dg})\n(running)', fontsize=9)
            continue
        ax.imshow(r['rho'].T, origin='lower', cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(f'η={E}h c={C} V({Dg},{Dg})\n{r["evals"]} ev, {r["wall"]:.0f} s, f {r["f"]:.4f}', fontsize=9)
ax = fig.add_subplot(gs[len(ETAS), :])
labels, walls, cols = [], [], []
for E in ETAS:
    for C in CS:
        for Dg in DS:
            r = res[E, C, Dg]
            labels.append(f'{E}h\nc{C}\nV{Dg}')
            walls.append(r['wall'] if r and r['done'] else np.nan)
            cols.append({2: '#40004b', 1: '#9970ab'}[Dg])
ax.bar(range(len(walls)), walls, color=cols)
ax.set_xticks(range(len(walls)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('laptop wall time [s]')
ax.set_title('time to the converged design (dark: V(2,2), light: V(1,1))')
ax.grid(alpha=0.3, axis='y')
fig.suptitle(f'Settings prototype, 2D TO {N}², contrast 1e{args.soft_exp}, CH start bi_s11, MG + Helmholtz L-BFGS', fontsize=12)
fig.tight_layout()
out = HERE / 'figures' / f'proto_N{N}_soft{args.soft_exp}.png'
fig.savefig(out, dpi=170)
print(out)
