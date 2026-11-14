"""
Scaling study of the multigrid preconditioner vs Green / Green-Jacobi / dual (NEMO, one GPU or laptop CPU).

Problem: the phase-field TO design of phasefield_twogrid.py at grid N with interface width eta (in pixels of N),
contrast k = 1e-4, macro strain E_xx.  Geometry = the 256^2 design (data/) thresholded at 0.5;
N > 256: pixel-replicated (identical geometry), N < 256: block-averaged then thresholded (as in the laptop study).
    phi = 1/2 [1 + tanh(d / (2 eta))] on the nodes, d = periodic signed distance in pixels of N
    phi_q = mean of the three nodes of each triangle (= muFFTTO N operator at the centroid, --check-N verifies)
    s = phi_q^2 (1 - k) + k;   eta = 0: s = 1 if phi_q > 0.5 else k (two-phase at the quadrature points)
    A = s C_0 (scalar material: dual material C_0 / s, low memory)
Solver: library CG, energy stop (relative 1e-6, adaptive delay tau 0.25, minimal delay --d-min; the NEMO scaling
study uses d_min = 0 by the user's decision 2026-11-13: the floor of 10 doubled MG's cost), no reference solution.

Output: <out>/csv/mg_scaling_N<N>.csv, one row per (eta, method).  Rows already present are skipped (restart path).

    python scaling_case.py --grid 1024 --eta 0.5 --methods mg green dual green_jacobi --device gpu --out DIR
"""
import sys
import os
import argparse
import csv
import socket
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import netCDF4
import numpy as np
from scipy import ndimage

from muFFTTO import solvers

import mg_lib
import problems

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--grid', type=int, required=True)
parser.add_argument('--eta', type=float, required=True, help='interface width eta / h')
parser.add_argument('--methods', nargs='+', default=['mg', 'green', 'dual', 'green_jacobi'])
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
parser.add_argument('--out', default=str(HERE / 'exp_data' / 'scaling'))
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--d-min', type=int, default=10)
parser.add_argument('--maxiter', type=int, default=100000)
parser.add_argument('--check-N', action='store_true', help='compare phi_q with the muFFTTO N operator (small N)')
args = parser.parse_args()

N, eta, k_soft = args.grid, args.eta, 1e-4
csv_dir = Path(args.out) / 'csv'
csv_dir.mkdir(parents=True, exist_ok=True)
csv_file = csv_dir / f'mg_scaling_N{N}.csv'
FIELDS = ['N', 'eta_over_h', 'method', 'iterations', 'stop_iteration', 'accepted_iterate', 'cost_K', 'cost_G', 'cost_T', 'work',
          't_setup', 't_solve', 't_per_it', 'mem_GB', 's_min', 'levels', 'd_min', 'device', 'host', 'job']
done = set()
if csv_file.exists():
    with open(csv_file) as f:
        done = {(float(r['eta_over_h']), r['method']) for r in csv.DictReader(f)}
todo = [m for m in args.methods if (eta, m) not in done]
if not todo:
    print(f'N {N} eta/h {eta:g}: all methods done, skip', flush=True)
    sys.exit(0)

if args.device == 'gpu':
    import cupy

    def sync():
        cupy.cuda.Device().synchronize()

    def mem_used_GB():
        free, total = cupy.cuda.runtime.memGetInfo()
        return (total - free) / 2 ** 30
else:
    def sync():
        pass

    def mem_used_GB():
        return float('nan')

# ---------------------------------------------------------------- geometry and material (host)
t0 = time.time()
design = np.asarray(netCDF4.Dataset(HERE / 'data' / 'N256_soft0.0001_energy-warm_Green_eta_0.0078125_w_5.0_l_bfgs_final.nc')
                    ['phase_field_solution'][:]).squeeze()
if N < 256:
    f = 256 // N
    solid = design.reshape(N, f, N, f).mean(axis=(1, 3)) > 0.5
else:
    solid = np.kron(design > 0.5, np.ones((N // 256, N // 256), dtype=bool))
if eta == 0:
    phi = solid.astype(float)
else:
    big = np.tile(solid, (3, 3))
    d_in = ndimage.distance_transform_edt(big)[N:2 * N, N:2 * N]
    d_out = ndimage.distance_transform_edt(~big)[N:2 * N, N:2 * N]
    del big
    phi = 0.5 * (1 + np.tanh(np.where(solid, d_in - 0.5, -(d_out - 0.5)) / (2 * eta)))
    del d_in, d_out
phi_x, phi_y = np.roll(phi, -1, axis=0), np.roll(phi, -1, axis=1)
phi_q = np.stack([(phi + phi_x + phi_y) / 3, (phi_x + phi_y + np.roll(phi_x, -1, axis=1)) / 3])   # [q, N, N]
s_q = np.where(phi_q > 0.5, 1.0, k_soft) if eta == 0 else phi_q ** 2 * (1 - k_soft) + k_soft
t_geom = time.time() - t0

# ---------------------------------------------------------------- hierarchy (all levels) and system
t0 = time.time()
hier = mg_lib.Hierarchy(s_q, problems.C_0, n_coarsest=4, device=args.device, scalar=True)
fine = hier.levels[0]
disc = fine.disc
xp = fine.xp
if args.check_N:
    ph = disc.get_scalar_field(name='phase')
    phq = disc.get_quad_field_scalar(name='phase_q')
    ph.s[0, 0] = xp.asarray(phi)
    disc.apply_N_operator_mugrid(ph, phq)
    print(f'check N operator: max |phi_q - N phi| = {float(abs(mg_lib.to_host(phq.s[0, 0]) - phi_q).max()):.1e}')
macro = disc.get_gradient_size_field(name='macro_gradient_field')
disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                     macro_gradient_field_ijqxyz=macro)
rhs = disc.get_unknown_size_field(name='rhs')
disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
x = disc.get_unknown_size_field(name='solution')
sync()
t_build = time.time() - t0
print(f'N {N} eta/h {eta:g}: s in [{s_q.min():.2e}, {s_q.max():.2e}], levels {[lv.N for lv in hier.levels]}, '
      f'geometry {t_geom:.1f} s, build {t_build:.1f} s, device {args.device}, mem {mem_used_GB():.2f} GB', flush=True)


def K_fun(xf, Axf):
    fine._apply_K(fine.material, xf, Axf)


def reset_caches():
    """every method pays its own setup (Jacobi diagonals, spectral estimates, coarse inverse)"""
    for lev in hier.levels:
        lev._dhalf = lev._dinv = None
        lev._lmax = {}
        lev._material_dual = None
    hier._pinv = None


new_file = not csv_file.exists()
for method in todo:
    reset_caches()
    sync()
    t0 = time.time()
    hier.setup(method)
    sync()
    t_setup = time.time() - t0
    B = hier.preconditioner(method)

    def P_fun(r, z):
        z.s[...] = B(r.s)

    hist = {'it': 0}

    def count(it, x_s, r_s, p_s, z_s, stop_crit):
        hist['it'] = it

    x.s[...] = 0.0
    hier.cost.update({'K': 0.0, 'G': 0.0, 'T': 0.0})
    sync()
    t0 = time.time()
    out = solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=x, P=P_fun, tol=args.tol,
        maxiter=args.maxiter, callback=count, stop_crit_norm='energy_lower_estim', energy_rtol=True,
        d_min=args.d_min, tau=0.25)
    sync()
    t_solve = time.time() - t0
    norms = out[1] if isinstance(out, tuple) else {}
    stop_it = norms.get('stop_iteration', {}).get('energy_lower_estim', -1) if isinstance(norms, dict) else -1
    # iterate l whose (delayed) estimate triggered the stop: k = l + delay -> reference-free stand-in for k*
    accepted = len(norms.get('energy_lower_estim', [])) - 1 if isinstance(norms, dict) else -1
    c = dict(hier.cost)
    row = {'N': N, 'eta_over_h': eta, 'method': method, 'iterations': hist['it'], 'stop_iteration': stop_it,
           'accepted_iterate': accepted,
           'cost_K': f'{c["K"]:.3f}', 'cost_G': f'{c["G"]:.3f}', 'cost_T': f'{c["T"]:.3f}',
           'work': f'{c["K"] + 0.2 * c["G"] + 0.05 * c["T"]:.2f}', 't_setup': f'{t_setup:.3f}',
           't_solve': f'{t_solve:.3f}', 't_per_it': f'{t_solve / max(hist["it"], 1):.5f}',
           'mem_GB': f'{mem_used_GB():.2f}', 's_min': f'{s_q.min():.3e}', 'levels': len(hier.levels),
           'd_min': args.d_min,
           'device': args.device, 'host': socket.gethostname(), 'job': os.environ.get('SLURM_JOB_ID', '-')}
    with open(csv_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            new_file = False
        w.writerow(row)
    print(f'N {N} eta/h {eta:g} {method:13s} it {hist["it"]:6d} (l {accepted})  work {row["work"]:>9s}  '
          f'setup {t_setup:7.1f} s  solve {t_solve:8.1f} s  ({t_solve / max(hist["it"], 1) * 1e3:.1f} ms/it)  '
          f'mem {row["mem_GB"]} GB', flush=True)
