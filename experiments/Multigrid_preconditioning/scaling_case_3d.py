"""
3D scaling of the multigrid preconditioner (trilinear hexahedra, option (a) re-discretised coarse grids) vs Green /
Green-Jacobi / dual, on the new random 3D Cahn-Hilliard geometries (make_new_CH_geometries.py, data/CH3D/).

Geometry: ch3d_<label>.npy (128^3, 1 = stiff); N < 128: block-averaged and thresholded; N = 128: as is.
    phi = 1/2 [1 + tanh(d / (2 eta))] on the nodes, d = periodic signed distance in pixels of N (eta = 0: sharp)
    phi_q = muFFTTO N operator (Gauss points), s = phi_q^2 (1 - k) + k (eta = 0: 1 / k by phi_q > 0.5), k = 1e-4
    A = s C_0 (K = 1, mu = 1/2), load E_xx
Solver: library CG, energy stop rtol 1e-6, adaptive delay with d_min (default 0 as in the 2D scaling study).
Output: <out>/csv/mg3d_scaling_<label>_N<N>.csv, one row per (eta, method); present rows are skipped (restart).

    python scaling_case_3d.py --geometry ch_s12 --grid 64 --eta 2 --methods mg green_jacobi green dual [--device gpu]
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

import numpy as np
from scipy import ndimage

from muFFTTO import material_models, solvers

import mg_lib

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--geometry', default='ch_s12')
parser.add_argument('--grid', type=int, required=True)
parser.add_argument('--eta', type=float, required=True, help='interface width eta / h (0: sharp)')
parser.add_argument('--methods', nargs='+', default=['mg', 'green_jacobi', 'green', 'dual'])
parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
parser.add_argument('--out', default=str(HERE / 'exp_data' / 'scaling3d'))
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--d-min', type=int, default=0)
parser.add_argument('--maxiter', type=int, default=50000)
parser.add_argument('--element', default='trilinear_hexahedron',
                    choices=['trilinear_hexahedron', 'trilinear_hexahedron_1Q', 'trilinear_hexahedron_1Q_hg'])
parser.add_argument('--hg-eps', type=float, default=0.1,
                    help='trilinear_hexahedron_1Q_hg: hourglass control as a fraction of the fully integrated stiffness')
args = parser.parse_args()

N, eta, k_soft = args.grid, args.eta, 1e-4
csv_dir = Path(args.out) / 'csv'
csv_dir.mkdir(parents=True, exist_ok=True)
etag = {'trilinear_hexahedron': '', 'trilinear_hexahedron_1Q': '_1Q',
        'trilinear_hexahedron_1Q_hg': f'_1Qhg{args.hg_eps:g}'}[args.element]
csv_file = csv_dir / f'mg3d_scaling_{args.geometry}_N{N}{etag}.csv'
FIELDS = ['geometry', 'element', 'N', 'eta_over_h', 'method', 'iterations', 'accepted_iterate', 'C11', 'cost_K', 'cost_G', 'cost_T',
          'work', 't_setup', 't_solve', 't_per_it', 'mem_GB', 's_min', 'levels', 'd_min', 'device', 'host', 'job']
done = set()
if csv_file.exists():
    with open(csv_file) as f:
        done = {(float(r['eta_over_h']), r['method']) for r in csv.DictReader(f)}
todo = [m for m in args.methods if (eta, m) not in done]
if not todo:
    print(f'{args.geometry} N {N} eta/h {eta:g}: all methods done, skip', flush=True)
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

C_0 = material_models.get_elastic_material_tensor(dim=3, K=1.0, mu=0.5, kind='linear')

# ---------------------------------------------------------------- geometry (host)
t0 = time.time()
phase = np.load(HERE / 'data' / 'CH3D' / f'ch3d_{args.geometry}.npy').astype(float)
M = phase.shape[0]
if N < M:
    f = M // N
    solid = phase.reshape(N, f, N, f, N, f).mean(axis=(1, 3, 5)) > 0.5
elif N == M:
    solid = phase > 0.5
else:
    f = N // M
    solid = np.kron(phase > 0.5, np.ones((f, f, f), dtype=bool))
if eta == 0:
    phi = solid.astype(float)
else:
    big = np.tile(solid, (3, 3, 3))
    d_in = ndimage.distance_transform_edt(big)[N:2 * N, N:2 * N, N:2 * N]
    d_out = ndimage.distance_transform_edt(~big)[N:2 * N, N:2 * N, N:2 * N]
    del big
    phi = 0.5 * (1 + np.tanh(np.where(solid, d_in - 0.5, -(d_out - 0.5)) / (2 * eta)))
    del d_in, d_out
t_geom = time.time() - t0

# ---------------------------------------------------------------- hierarchy, material via the muFFTTO N operator
t0 = time.time()
nq = 1 if '_1Q' in args.element else 8
hier = mg_lib.Hierarchy(np.ones((nq, N, N, N)), C_0, n_coarsest=4, device=args.device, scalar=True,
                        element_type=args.element, hg_eps=args.hg_eps)
fine = hier.levels[0]
disc = fine.disc
xp = fine.xp
ph = disc.get_scalar_field(name='phase')
phq = disc.get_quad_field_scalar(name='phase_q')
ph.s[0, 0] = xp.asarray(phi)
disc.apply_N_operator_mugrid(ph, phq)
rq = mg_lib.to_host(phq.s[0, 0])
s_q = np.where(rq > 0.5, 1.0, k_soft) if eta == 0 else rq ** 2 * (1 - k_soft) + k_soft
hier.update_material(s_q)
macro = disc.get_gradient_size_field(name='macro_gradient_field')
E = np.zeros((3, 3))
E[0, 0] = 1.0
disc.get_macro_gradient_field_mugrid(macro_gradient_ij=E, macro_gradient_field_ijqxyz=macro)
rhs = disc.get_unknown_size_field(name='rhs')
disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro, rhs_inxyz=rhs)
x = disc.get_unknown_size_field(name='solution')
sync()
t_build = time.time() - t0
print(f'{args.geometry} N {N} eta/h {eta:g}: stiff {solid.mean():.2f}, s in [{s_q.min():.2e}, {s_q.max():.2e}], '
      f'levels {[lv.N for lv in hier.levels]}, geometry {t_geom:.1f} s, build {t_build:.1f} s, mem {mem_used_GB():.2f} GB',
      flush=True)


def K_fun(xf, Axf):
    fine._apply_K(fine.material, xf, Axf)


def reset_caches():
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
    accepted = len(norms.get('energy_lower_estim', [])) - 1 if isinstance(norms, dict) else -1
    sig = disc.get_homogenized_stress_mugrid(material_data_field_ijklqxyz=fine.material, displacement_field_inxyz=x,
                                             macro_gradient_field_ijqxyz=macro, formulation='small_strain')
    C11 = float(np.asarray(mg_lib.to_host(sig))[0, 0])
    c = dict(hier.cost)
    row = {'geometry': args.geometry, 'element': args.element, 'N': N, 'eta_over_h': eta, 'method': method,
           'iterations': hist['it'], 'accepted_iterate': accepted, 'C11': f'{C11:.8f}', 'cost_K': f'{c["K"]:.3f}', 'cost_G': f'{c["G"]:.3f}', 'cost_T': f'{c["T"]:.3f}',
           'work': f'{c["K"] + 0.2 * c["G"] + 0.05 * c["T"]:.2f}', 't_setup': f'{t_setup:.3f}',
           't_solve': f'{t_solve:.3f}', 't_per_it': f'{t_solve / max(hist["it"], 1):.5f}',
           'mem_GB': f'{mem_used_GB():.2f}', 's_min': f'{s_q.min():.3e}', 'levels': len(hier.levels),
           'd_min': args.d_min, 'device': args.device, 'host': socket.gethostname(),
           'job': os.environ.get('SLURM_JOB_ID', '-')}
    with open(csv_file, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            new_file = False
        w.writerow(row)
    print(f'{args.geometry}{etag} N {N} eta/h {eta:g} {method:13s} it {hist["it"]:6d} (l {accepted})  C11 {C11:.6f}  work {row["work"]:>10s}  '
          f'setup {t_setup:7.1f} s  solve {t_solve:8.1f} s  ({t_solve / max(hist["it"], 1) * 1e3:.1f} ms/it)  '
          f'mem {row["mem_GB"]} GB', flush=True)
