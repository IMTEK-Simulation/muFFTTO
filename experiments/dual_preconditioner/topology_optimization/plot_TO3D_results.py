"""3D TO results from NEMO (example_3D_elasticity_TO.py, contrast 1e4, w = 0.2): designs and preconditioner convergence.

Data (copied from NEMO):  data/example_3D_elasticity_TO/nemo/N<N>_..._<precond>_eta_<eta>_..._l_bfgs<suffix>[_seg<k>]_{final,history}.nc
Logs:                     logs/TO3D_nemo/TO3D_N<N>_eta<eta>_<precond><suffix>[_seg<k>].log
Figures:                  figures/TO3D_designs_eta<k>h.png, figures/TO3D_convergence.png

Two series: eta = 4h (64^3 eta 1/16, 128^3 1/32, 192^3 1/48, 256^3 1/64; the rule of NOTES_eta_h_choice.md; 192^3 and
256^3 started from the 128^3 designs, suffix _from128) and eta = 2h (32^3 1/16, 64^3 1/32: under-resolved interface).
A run consists of segments of at most 500 L-BFGS iterations (sbatch_TO3D_gpu.sh), each restarted from the design of
the previous one; they are concatenated here (iteration counted over all segments). Running runs have no _final.nc
yet: their last _history.nc frame is shown (every 10th iterate is stored).
Cost: CG iterations of the state + adjoint solves. One dual CG iteration costs two Green / Green-Jacobi ones (two
FFT pairs), so 'work' counts dual iterations twice.
"""
import glob
import os
import re

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

from elastic_properties import properties, voigt_from_stresses

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'example_3D_elasticity_TO', 'nemo')
LOGS = os.path.join(HERE, 'logs', 'TO3D_nemo')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 13})

COLOR = {'Green': 'tab:green', 'Green_Jacobi': 'tab:blue', 'dual': 'tab:red'}
LABEL = {'Green': 'Green', 'Green_Jacobi': 'Green-Jacobi', 'dual': 'dual'}
PRECONDS = ('Green', 'Green_Jacobi', 'dual')
COST = {'Green': 1.0, 'Green_Jacobi': 1.0, 'dual': 2.0}
# series -> grids: (N, eta as in the file names, suffix)
SERIES = {'4h': [(64, '0.0625', ''), (128, '0.03125', ''), (192, '0.0208333333', '_from128'),
                 (256, '0.015625', '_from128')],
          '2h': [(32, '0.0625', ''), (64, '0.03125', '')]}
ETA_NAME = {'0.0625': '1/16', '0.03125': '1/32', '0.0208333333': '1/48', '0.015625': '1/64'}
LS = {32: ':', 64: '-', 128: '--', 192: '-.', 256: (0, (1, 1))}


def segments(N, P, eta, suffix):
    """Log files of a run in order: base, _seg2, _seg3, ..."""
    base = os.path.join(LOGS, f'TO3D_N{N}_eta{eta}_{P}{suffix}')
    files = [base + '.log'] if os.path.exists(base + '.log') else []
    k = 2
    while os.path.exists(f'{base}_seg{k}.log'):
        files.append(f'{base}_seg{k}.log')
        k += 1
    return files


def parse_log(N, P, eta, suffix):
    """Per L-BFGS iteration over all segments: iteration, objective, CG iterations (state + adjoint) and solves since
    the previous iteration line; converged: a segment reports L-BFGS convergence."""
    it, obj, cg, ns = [], [], [], []
    converged = False
    for f in segments(N, P, eta, suffix):
        offset = 0
        acc = solves = 0
        for line in open(f):
            m = re.match(r'start .* iterations_done=(\d+)', line)
            if m:
                offset = int(m.group(1))
                continue
            m = re.search(r'nb_ ?steps CG (?:mech|adjoint)\s*=\s*(\d+)', line)
            if m:
                acc += int(m.group(1))
                solves += 1
                continue
            m = re.match(r'^(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(\S+)\s*$', line)
            if m:
                if offset and int(m.group(1)) == 0 and it:   # restart: iteration 0 repeats the last iterate
                    cg[-1] += acc; ns[-1] += solves
                else:
                    it.append(offset + int(m.group(1))); obj.append(float(m.group(2)))
                    cg.append(acc); ns.append(solves)
                acc = solves = 0
            if 'OPTIMIZER success=True' in line:
                converged = True
            # runs from before the OPTIMIZER line: a segment that saved its data below the 500-iteration cap converged
            if line.startswith('Data saved') and it and it[-1] - offset < 500:
                converged = True
    if not it:
        return None
    return dict(it=np.array(it), obj=np.array(obj), cg=np.array(cg), ns=np.array(ns), converged=converged)


def load_design(N, P, eta, suffix):
    """Final design of the last finished segment, else the last stored frame of the newest history."""
    hist = sorted(glob.glob(os.path.join(DATA, f'N{N}_soft0.0001_energy-warm_{P}_eta_{eta}_pfw0.2_l_bfgs{suffix}'
                                               f'*_history.nc')), key=os.path.getmtime)
    hist = [h for h in hist if re.search(rf'l_bfgs{suffix}(_seg\d+)?_history\.nc$', h)]
    if not hist:
        return None
    s = hist[-1][:-len('_history.nc')]
    if os.path.exists(s + '_final.nc'):
        d = netCDF4.Dataset(s + '_final.nc')
        return np.array(d.variables['phase_field_solution'][0, 0])
    d = netCDF4.Dataset(s + '_history.nc')
    return np.array(d.variables['phase_field_iterate'][d.dimensions['frame'].size - 1, 0])


def final_properties(N, P, eta, suffix):
    """Effective properties (elastic_properties.properties) of the last finished segment, None while running."""
    hist = sorted(glob.glob(os.path.join(DATA, f'N{N}_soft0.0001_energy-warm_{P}_eta_{eta}_pfw0.2_l_bfgs{suffix}'
                                               f'*_final.nc')), key=os.path.getmtime)
    hist = [h for h in hist if re.search(rf'l_bfgs{suffix}(_seg\d+)?_final\.nc$', h)]
    if not hist:
        return None
    with netCDF4.Dataset(hist[-1]) as d:
        return properties(voigt_from_stresses(np.array(d.getncattr('homogenized_stresses')).reshape(6, 3, 3)))


def surface(ax, rho, title):
    N = rho.shape[0]
    step = max(1, N // 64)
    r = np.pad(1.0 - rho, 1, mode='constant', constant_values=0.0)    # soft (void) phase, closed at the cell boundary
    v, f, _, _ = measure.marching_cubes(r, 0.5, step_size=step)
    v = (v - 1) / N
    mesh = Poly3DCollection(v[f], linewidth=0)
    nrm = np.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0])
    shade = 0.35 + 0.65 * np.abs(nrm @ np.array([0.4, 0.5, 0.77])) / (np.linalg.norm(nrm, axis=1) + 1e-30)
    mesh.set_facecolor(plt.cm.Oranges(0.25 + 0.6 * shade))
    ax.add_collection3d(mesh)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), xticks=[], yticks=[], zticks=[])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=35)
    ax.set_title(title, fontsize=13)


def interface_area(rho):
    """Area of the rho = 0.5 surface per unit cell (periodic, cell side 1)."""
    N = rho.shape[0]
    v, f, _, _ = measure.marching_cubes(np.pad(rho, 1, mode='wrap'), 0.5)
    keep = np.all((v[f] >= 1) & (v[f] < N + 1), axis=(1, 2))           # one periodic copy
    return measure.mesh_surface_area(v, f[keep]) / N ** 2


def table():
    print(f'{"eta":>4} {"N":>4} {"eta":>5} {"precond":>13} {"it":>5} {"conv":>5} {"f":>8} {"CG/solve":>9} '
          f'{"work[k]":>8} {"grey%":>6} {"S":>5}  nu12 / nu13 / mu / A_Z (target -1/3 in 2D, 3D: -0.5 / -0.5 / 0.193 / 1)')
    for series, grids in SERIES.items():
        for N, eta, suffix in grids:
            for P in PRECONDS:
                lg = parse_log(N, P, eta, suffix)
                if lg is None:
                    continue
                rho = load_design(N, P, eta, suffix)
                per = lg['cg'][1:].sum() / max(lg['ns'][1:].sum(), 1)       # without the start (iteration 0)
                grey = np.mean((rho > 0.1) & (rho < 0.9)) if rho is not None else np.nan
                S = interface_area(rho) if rho is not None else np.nan
                pr = final_properties(N, P, eta, suffix)
                print(f'{series:>4} {N:4d} {ETA_NAME[eta]:>5} {P:>13} {lg["it"][-1]:5d} {str(lg["converged"]):>5} '
                      f'{lg["obj"][-1]:8.4f} {per:9.1f} {COST[P] * lg["cg"].sum() / 1e3:8.0f} {100 * grey:6.1f} '
                      f'{S:5.2f}  ' + ('-' if pr is None else
                                      f'{pr["nu12"]:+.3f} / {pr["nu13"]:+.3f} / {pr["mu"]:.3f} / {pr["A_Z"]:.2f}'))


def designs():
    """One preconditioner per grid (Green-Jacobi: run on every grid; the designs of the three agree, see table())."""
    for series, grids in SERIES.items():
        fig = plt.figure(figsize=(3.9 * len(grids), 7.8))
        for c, (N, eta, suffix) in enumerate(grids):
            P = 'Green_Jacobi'
            rho = load_design(N, P, eta, suffix)
            lg = parse_log(N, P, eta, suffix)
            ax3 = fig.add_subplot(2, len(grids), c + 1, projection='3d')
            ax2 = fig.add_subplot(2, len(grids), len(grids) + c + 1)
            state = ('converged' if lg['converged'] else 'running') + f', it {lg["it"][-1]}'
            surface(ax3, rho, f'{N}$^3$, $\\eta$ = {ETA_NAME[eta]}\n{state}')
            ax2.imshow(rho[:, :, N // 2].T, origin='lower', cmap='Greys', vmin=0, vmax=1,
                       interpolation='nearest', extent=(0, 1, 0, 1))
            ax2.set_title(f'slice z = 1/2, grey {100 * np.mean((rho > 0.1) & (rho < 0.9)):.0f} %', fontsize=12)
            ax2.set_xticks([0, 0.5, 1]); ax2.set_yticks([0, 0.5, 1])
            ax2.set_xlabel('x'); ax2.set_ylabel('y')
        fig.suptitle(f'3D TO designs, $\\eta$ = {series} (Green-Jacobi runs), contrast 1e4, w = 0.2\n'
                     f'top: soft phase $\\rho$ < 0.5;  bottom: $\\rho$ (black = stiff)', fontsize=14)
        fig.tight_layout()
        out = os.path.join(FIG, f'TO3D_designs_eta{series}.png')
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print('saved', out)


def convergence():
    fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
    for row, (series, grids) in enumerate(SERIES.items()):
        for N, eta, suffix in grids:
            for P in PRECONDS:
                lg = parse_log(N, P, eta, suffix)
                if lg is None or len(lg['it']) < 3:
                    continue
                kw = dict(color=COLOR[P], ls=LS[N], lw=2, label=f'{LABEL[P]} {N}$^3$')
                axs[row, 0].plot(lg['it'], lg['obj'], **kw)
                per = lg['cg'][1:] / np.maximum(lg['ns'][1:], 1)
                w = min(10, len(per))
                axs[row, 1].plot(lg['it'][w:], np.convolve(per, np.ones(w) / w, mode='valid'), **kw)
                axs[row, 2].plot(lg['it'], np.cumsum(lg['cg']) * COST[P] / 1e3, **kw)
        axs[row, 0].set(yscale='log', ylabel='objective f', title=f'$\\eta$ = {series}: objective')
        axs[row, 0].set_yticks([2, 3, 5, 10, 20, 30])
        axs[row, 0].yaxis.set_major_formatter(plt.ScalarFormatter())
        axs[row, 0].yaxis.set_minor_formatter(plt.NullFormatter())
        axs[row, 1].set(ylabel='CG iterations per solve', ylim=(0, None),
                        title=f'$\\eta$ = {series}: CG per solve (mean of 10 it.)')
        axs[row, 2].set(ylabel='cumulative work [10$^3$ CG it.]', ylim=(0, None),
                        title=f'$\\eta$ = {series}: work (dual counted $\\times$2)')
        for ax in axs[row]:
            ax.set_xlabel('L-BFGS iteration')
            ax.set_xlim(0, None)
            ax.grid(alpha=0.3)
        axs[row, 2].legend(fontsize=11, ncol=2, loc='upper left')
    fig.suptitle('3D TO convergence (contrast 1e4, w = 0.2); line style = grid', fontsize=15)
    fig.tight_layout()
    out = os.path.join(FIG, 'TO3D_convergence.png')
    fig.savefig(out, dpi=115)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    table()
    convergence()
    designs()
