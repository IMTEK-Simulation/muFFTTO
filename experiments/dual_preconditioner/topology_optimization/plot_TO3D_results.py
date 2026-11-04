"""3D TO results from NEMO (example_3D_elasticity_TO.py, contrast 1e4, w = 0.2): designs and convergence.

Data (copied from NEMO):  data/example_3D_elasticity_TO/nemo/N<N>_..._<precond>_eta_<eta>_..._{final,history}.nc
Logs:                     logs/TO3D_nemo/TO3D_N<N>_eta<eta>_<precond>.log
Figures:                  figures/TO3D_designs_eta<eta>.png, figures/TO3D_convergence.png

Running jobs have no _final.nc yet: their last _history.nc frame is shown (every 10th iterate is stored).
Homogenised tangent: built from the stored per-load-case stresses (the 'Optimized elastic tangent' printed by runs
started before the range(dim) fix is wrong in columns 3-6).
Cost: CG iterations of state + adjoint solves. One dual CG iteration costs about two Green / Green-Jacobi ones
(two FFT pairs), so the work panel counts dual iterations twice.
"""
import os
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'example_3D_elasticity_TO', 'nemo')
LOGS = os.path.join(HERE, 'logs', 'TO3D_nemo')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 13})

COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
LS = {32: ':', 64: '-', 128: '--'}
COST = {'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}
RUNS = {0.0625: [(32, 'Green'), (32, 'Green_Jacobi'), (32, 'dual'), (64, 'Green'), (64, 'Green_Jacobi'),
                 (64, 'dual')],
        0.03125: [(64, 'Green'), (64, 'Green_Jacobi'), (64, 'dual'), (128, 'Green_Jacobi'), (128, 'dual')]}
ETA_NAME = {0.0625: '1/16', 0.03125: '1/32'}


def stem(N, P, eta):
    return os.path.join(DATA, f'N{N}_soft0.0001_energy-warm_{P}_eta_{eta}_pfw0.2_l_bfgs')


def load_design(N, P, eta):
    """Final design if the run finished, else the last stored history frame. Returns (rho, label, attrs)."""
    s = stem(N, P, eta)
    if os.path.exists(s + '_final.nc'):
        d = netCDF4.Dataset(s + '_final.nc')
        rho = np.array(d.variables['phase_field_solution'][0, 0])
        attrs = {k: d.getncattr(k) for k in d.ncattrs()}
        return rho, f'final, {int(attrs["nb_iterations"])} it', attrs
    if os.path.exists(s + '_history.nc'):
        d = netCDF4.Dataset(s + '_history.nc')
        nf = d.dimensions['frame'].size
        rho = np.array(d.variables['phase_field_iterate'][nf - 1, 0])
        return rho, f'running, frame {nf}', {}
    return None, 'not available', {}


def parse_log(N, P, eta):
    """Per L-BFGS iteration: objective, CG iterations (state + adjoint) and number of solves since the previous one."""
    f = os.path.join(LOGS, f'TO3D_N{N}_eta{eta}_{P}.log')
    if not os.path.exists(f):
        return None
    it, obj, cg, ns = [], [], [], []
    acc = solves = 0
    for line in open(f):
        m = re.search(r'nb_ ?steps CG (?:mech|adjoint)\s*=\s*(\d+)', line)
        if m:
            acc += int(m.group(1))
            solves += 1
            continue
        m = re.match(r'^(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\S+)\s*$', line)
        if m:
            it.append(int(m.group(1)))
            obj.append(float(m.group(2)))
            cg.append(acc)
            ns.append(solves)
            acc = solves = 0
    return {k: np.array(v) for k, v in dict(it=it, obj=obj, cg=cg, ns=ns).items()}


def tangent_from_stresses(attrs):
    s = np.array(attrs['homogenized_stresses']).reshape(6, 3, 3)
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
    return np.array([[s[k][i, j] for k in range(6)] for (i, j) in idx])


def surface(ax, rho, title):
    N = rho.shape[0]
    step = 1 if N <= 64 else 2
    r = np.pad(1.0 - rho, 1, mode='constant', constant_values=0.0)    # soft (void) phase, closed at the cell boundary
    v, f, _, _ = measure.marching_cubes(r, 0.5, step_size=step)
    v = (v - 1) / N
    mesh = Poly3DCollection(v[f], linewidth=0)
    light = np.abs(np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0]))
    shade = 0.35 + 0.65 * (light @ np.array([0.4, 0.5, 0.77]) / (np.linalg.norm(light, axis=1) + 1e-30))
    mesh.set_facecolor(plt.cm.Oranges(0.25 + 0.6 * shade))
    ax.add_collection3d(mesh)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xticks=[], yticks=[], zticks=[])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)
    ax.set_title(title, fontsize=13)


def area_of_interface(rho):
    """Surface area of the 0.5 level set per unit cell (periodic, unit cell side 1)."""
    N = rho.shape[0]
    r = np.pad(rho, 1, mode='wrap')
    v, f, _, _ = measure.marching_cubes(r, 0.5)
    keep = np.all((v[f] >= 1) & (v[f] < N + 1), axis=(1, 2))           # one periodic copy
    return measure.mesh_surface_area(v, f[keep]) / N ** 2


def designs():
    table = []
    for eta, runs in RUNS.items():
        fig = plt.figure(figsize=(3.4 * len(runs), 7.4))
        for c, (N, P) in enumerate(runs):
            rho, state, attrs = load_design(N, P, eta)
            lg = parse_log(N, P, eta)
            if state.startswith('running') and lg is not None and len(lg['it']):
                state = f'running, it {lg["it"][-1]}'
            ax3 = fig.add_subplot(2, len(runs), c + 1, projection='3d')
            ax2 = fig.add_subplot(2, len(runs), len(runs) + c + 1)
            title = f'{LABEL[P]}, {N}$^3$\n{state}'
            if rho is None:
                ax3.set_axis_off(); ax2.set_axis_off(); ax3.set_title(title); continue
            surface(ax3, rho, title)
            ax3.title.set_color(COLOR[P])
            ax2.imshow(rho[:, :, N // 2].T, origin='lower', cmap='Greys', vmin=0, vmax=1,
                       interpolation='nearest', extent=(0, 1, 0, 1))
            grey = np.mean((rho > 0.1) & (rho < 0.9))
            ax2.set_title(f'slice z = 1/2   grey {100 * grey:.0f} %', fontsize=12)
            ax2.set_xticks([0, 0.5, 1]); ax2.set_yticks([0, 0.5, 1])
            ax2.set_xlabel('x'); ax2.set_ylabel('y')
            row = dict(N=N, P=P, eta=eta, state=state, vol=rho.mean(), grey=grey, S=area_of_interface(rho),
                       f=lg['obj'][-1] if lg is not None and len(lg['obj']) else np.nan,
                       it=lg['it'][-1] if lg is not None and len(lg['it']) else -1)
            if 'homogenized_stresses' in attrs:
                C = tangent_from_stresses(attrs)
                Ct = np.array(attrs['target_C_voigt']).reshape(6, 6)
                row['Cerr'] = np.linalg.norm(C - Ct) / np.linalg.norm(Ct)
                row['C11 C12 C44'] = (np.mean(np.diag(C)[:3]), np.mean([C[0, 1], C[0, 2], C[1, 2]]),
                                      np.mean(np.diag(C)[3:]))
            table.append(row)
        fig.suptitle(f'3D TO designs, $\\eta$ = {ETA_NAME[eta]}, contrast 1e4, w = 0.2  (top: void phase $\\rho$ < 0.5; bottom: $\\rho$, black = solid)',
                     fontsize=15)
        fig.tight_layout()
        out = os.path.join(FIG, f'TO3D_designs_eta{eta}.png')
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print('saved', out)
    print(f'{"N":>4} {"precond":>13} {"eta":>8} {"it":>4} {"f":>8} {"vol":>6} {"grey%":>6} {"S":>6} {"Cerr":>6}  '
          f'C11 C12 C44 (target 0.289 -0.096 0.193)')
    for r in table:
        c = ' '.join(f'{x:.3f}' for x in r['C11 C12 C44']) if 'C11 C12 C44' in r else '-'
        print(f'{r["N"]:4d} {r["P"]:>13} {r["eta"]:8.5f} {r["it"]:4d} {r["f"]:8.4f} {r["vol"]:6.3f} '
              f'{100 * r["grey"]:6.1f} {r["S"]:6.2f} {r.get("Cerr", np.nan):6.3f}  {c}')


def convergence():
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    for row, (eta, runs) in enumerate(RUNS.items()):
        for N, P in runs:
            lg = parse_log(N, P, eta)
            if lg is None or len(lg['it']) < 12:
                continue
            kw = dict(color=COLOR[P], ls=LS[N], lw=2, label=f'{LABEL[P]} {N}$^3$')
            axs[row, 0].plot(lg['it'], lg['obj'], **kw)
            ok = lg['ns'] > 0
            per = np.where(ok, lg['cg'] / np.maximum(lg['ns'], 1), np.nan)
            w = 10
            smooth = np.convolve(np.nan_to_num(per), np.ones(w) / w, mode='valid')
            axs[row, 1].plot(lg['it'][w - 1:], smooth, **kw)
            axs[row, 2].plot(lg['it'], np.cumsum(lg['cg']) * COST[P] / 1e3, **kw)
        axs[row, 0].set(yscale='log', ylabel='objective f', title=f'$\\eta$ = {ETA_NAME[eta]}: objective')
        axs[row, 1].set(ylabel='CG iterations per solve', ylim=(0, None),
                        title=f'$\\eta$ = {ETA_NAME[eta]}: CG per solve (mean of 10 it.)')
        axs[row, 2].set(ylabel='cumulative work [10$^3$ CG it.]', ylim=(0, None),
                        title=f'$\\eta$ = {ETA_NAME[eta]}: work (dual counted $\\times$2)')
        axs[row, 0].yaxis.set_major_formatter(plt.ScalarFormatter())
        axs[row, 0].yaxis.set_minor_formatter(plt.ScalarFormatter())
        for ax in axs[row]:
            ax.set_xlabel('L-BFGS iteration')
            ax.set_xlim(0, None)
            ax.grid(alpha=0.3)
    h, l = axs[1, 0].get_legend_handles_labels()
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    seen = dict(zip(l0 + l, h0 + h))
    fig.legend(seen.values(), seen.keys(), loc='lower center', ncol=6, fontsize=13, frameon=False)
    fig.suptitle('3D TO convergence (contrast 1e4, w = 0.2); line style: grid 32$^3$ dotted, 64$^3$ solid, '
                 '128$^3$ dashed', fontsize=15)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = os.path.join(FIG, 'TO3D_convergence.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    convergence()
    designs()
