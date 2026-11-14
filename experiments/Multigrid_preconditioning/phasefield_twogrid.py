"""
User's two-grid idea on a phase-field geometry (2026-11-13):
    "run a couple of Green-CG iterations on 256 to remove the high-frequency error, go to the coarse grid 128,
     run again a couple of Green-CG iterations, go back to 256 -- different frequencies of the error are reduced."

Geometry: the 256^2 TO design of experiments/dual_preconditioner/topology_optimization/phasefield_interface_study.py
(nemo_eta2h, soft 1e-4; copied to data/), thresholded and given the tanh phase-field profile of width eta:
    phi = 1/2 [1 + tanh(d / (2 eta))],  s = phi_q^2 (1 - k) + k,  k = 1e-4,  A = s C_0,  load E_xx.
Reference counts there (library CG, energy stop): eta = 2h: Green 571, GJ 162, dual 201.

Cycle (green2grid nu nu_c):  x <- nu Green-PCG iterations on 256 from the current x  (restarted CG)
                             x <- x + P [nu_c Green-PCG iterations on the 128 Galerkin problem for R (b - A x)]
repeated until the true relative energy error <= tol^2.  Coarse operator = Galerkin (material averaged over the
4 fine triangles, mg_lib).  Baselines with the same local PCG: Green, GJ, dual, MG (V-cycle, Chebyshev-Jacobi),
MG with Chebyshev-Green smoother (mgS-green).  All iterations are counted to the TRUE error (k*), no delay.

Diagnostic: radially binned error spectrum  E(k) = sum_{|xi| in shell k} |xi|^2 |e_hat(xi)|^2  (strain-energy
weighted; numpy.fft only for this offline analysis) after the stages of the first cycle, as a ratio to the
initial error -- shows which frequencies each stage removes.

    python phasefield_twogrid.py [--eta 2] [--cycles 2,2 5,5 5,20]
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import netCDF4
import numpy as np
from scipy import ndimage

from muFFTTO import domain

import mg_lib
import problems

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--eta', type=float, default=2.0, help='interface width eta / h')
parser.add_argument('--grid', type=int, default=256, help='N; N < 256: the 256^2 design block-averaged, then thresholded')
parser.add_argument('--cycles', nargs='*', default=['2,2', '5,5', '5,20'], help='nu,nu_c pairs (none: skip)')
parser.add_argument('--methods', nargs='*', default=['green', 'green_jacobi', 'dual', 'mg', 'mgS-green'])
parser.add_argument('--tag', default='', help='suffix of the output files')
parser.add_argument('--tol', type=float, default=1e-6, help='relative energy error (squared: tol^2)')
parser.add_argument('--w-G', type=float, default=0.2)
parser.add_argument('--w-T', type=float, default=0.05)
parser.add_argument('--max-work', type=float, default=3000.0)
parser.add_argument('--spectra-only', action='store_true',
                    help='skip the convergence runs; only the error spectra of the first stages')
args = parser.parse_args()

N, k_soft, p_simp = args.grid, 1e-4, 2
DESIGN = HERE / 'data' / 'N256_soft0.0001_energy-warm_Green_eta_0.0078125_w_5.0_l_bfgs_final.nc'

# ---------------------------------------------------------------- phase-field material at the quadrature points
design = np.asarray(netCDF4.Dataset(DESIGN)['phase_field_solution'][:]).squeeze()
if N < 256:                                               # same geometry, coarser resolution
    f = 256 // N
    design = design.reshape(N, f, N, f).mean(axis=(1, 3))
solid = design > 0.5
big = np.tile(solid, (3, 3))
d_in = ndimage.distance_transform_edt(big)[N:2 * N, N:2 * N]
d_out = ndimage.distance_transform_edt(~big)[N:2 * N, N:2 * N]
signed = np.where(solid, d_in - 0.5, -(d_out - 0.5))
phi = solid.astype(float) if args.eta == 0 else 0.5 * (1 + np.tanh(signed / (2 * args.eta)))
cell = domain.PeriodicUnitCell(domain_size=[1, 1], problem_type='elasticity')
disc0 = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N), discretization_type='finite_element',
                              element_type='linear_triangles')
phase = disc0.get_scalar_field(name='phase')
phase_q = disc0.get_quad_field_scalar(name='phase_q')
phase.s[0, 0] = phi
disc0.apply_N_operator_mugrid(phase, phase_q)
rq = np.asarray(phase_q.s[0, 0])
if args.eta == 0:                                         # sharp: two-phase at the quadrature points
    s_q = np.where(rq > 0.5, 1.0, k_soft)
else:
    s_q = np.power(rq, p_simp) * (1 - k_soft) + k_soft    # [q, N, N]
C_q = np.einsum('ijkl,qxy->ijklqxy', problems.C_0, s_q)
print(f'eta/h = {args.eta}, s in [{s_q.min():.2e}, {s_q.max():.2e}]', flush=True)

hier = mg_lib.Hierarchy(C_q, problems.C_0, n_coarsest=4)
fine, coarse = hier.levels[0], hier.levels[1]
macro = fine.disc.get_gradient_size_field(name='macro_gradient_field')
fine.disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                          macro_gradient_field_ijqxyz=macro)
rhs_f = fine.disc.get_unknown_size_field(name='rhs')
fine.disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro,
                         rhs_inxyz=rhs_f)
b = rhs_f.s.copy()


def work():
    c = hier.cost
    return c['K'] + args.w_G * c['G'] + args.w_T * c['T']


def reset():
    hier.cost.update({'K': 0.0, 'G': 0.0, 'T': 0.0})


def pcg(lev, rhs, x, M, iters, monitor=None, rz_rtol=None):
    """plain Hestenes-Stiefel PCG (local copy; the library CG is not restartable from inside a cycle)"""
    r = rhs.copy() if not np.any(x) else rhs - lev.K(x)
    z = M(r)
    p = z.copy()
    rz = rz0 = np.sum(r * z)
    for _ in range(iters):
        if rz_rtol is not None and rz < rz_rtol * rz0:
            break
        Ap = lev.K(p)
        alpha = rz / np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        if monitor is not None and monitor(x):
            break
        z = M(r)
        rz_new = np.sum(r * z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


# ---------------------------------------------------------------- reference solution (dual, tight)
t0 = time.time()
hier.setup('dual')
u_ref = pcg(fine, b, np.zeros_like(b), lambda r: fine.M('dual', r), 20000, rz_rtol=1e-28)
E0 = np.sum(u_ref * fine.K(u_ref, count=False))
print(f'reference: {time.time() - t0:.0f} s, |r|/|b| = '
      f'{np.linalg.norm(b - fine.K(u_ref, count=False)) / np.linalg.norm(b):.1e}', flush=True)


def rel_error(x):
    e = u_ref - x
    return np.sum(e * fine.K(e, count=False)) / E0


class Monitor:
    def __init__(self):
        self.work, self.error, self.it = [0.0], [1.0], 0
        reset()

    def __call__(self, x):
        self.it += 1
        self.work.append(work())
        self.error.append(rel_error(x))
        return self.error[-1] <= args.tol ** 2 or self.work[-1] > args.max_work


# ---------------------------------------------------------------- error spectrum
kx = np.fft.fftfreq(N, 1.0 / N)
KX, KY = np.meshgrid(kx, kx, indexing='ij')
kabs = np.sqrt(KX ** 2 + KY ** 2)
shell = np.rint(kabs).astype(int)
n_shell = N // 2 + 1


L_T = np.linalg.cholesky(mg_lib.to_mandel(problems.C_0)).T             # eps^T C_0 eps = |L^T eps|^2 (Mandel)


def spectrum_geometric(x):
    """|xi|^2 |e_hat|^2 per shell: displacement-gradient spectrum, every region weighted equally"""
    e = (u_ref - x)[:, 0]
    eh = np.fft.fft2(e, axes=(-2, -1))
    dens = (kabs ** 2 * np.abs(eh) ** 2).sum(axis=0)
    return np.bincount(shell.ravel(), weights=dens.ravel(), minlength=n_shell)[:n_shell]


def spectrum(x):
    """energy spectrum: f_q = sqrt(s_q) L^T eps_q(e) per triangle q, sum over shells of |f_hat|^2 = ||e||_K^2
    exactly (Parseval) -- the error measured in the norm the solvers minimise"""
    e = (u_ref - x)[:, 0]                                                # [2, N, N]
    ex, ey = np.roll(e, -1, axis=-2), np.roll(e, -1, axis=-1)
    exy = np.roll(ex, -1, axis=-1)
    grads = [np.stack([ex - e, ey - e], axis=1),                         # lower: d/dx, d/dy  [i, j, N, N]
             np.stack([exy - ey, exy - ex], axis=1)]                     # upper
    dens = 0.0
    for q, g in enumerate(grads):
        eps = np.array([g[0, 0], g[1, 1], np.sqrt(2.0) * 0.5 * (g[0, 1] + g[1, 0])])
        f = np.sqrt(s_q[q]) * np.einsum('ab,bxy->axy', L_T, eps)
        dens = dens + (np.abs(np.fft.fft2(f, axes=(-2, -1))) ** 2).sum(axis=0)
    return np.bincount(shell.ravel(), weights=dens.ravel(), minlength=n_shell)[:n_shell]


results = {}
spectra = {'initial': spectrum(np.zeros_like(b))}
spectra_geo = {'initial': spectrum_geometric(np.zeros_like(b))}
out = HERE / 'exp_data' / f'phasefield_twogrid_N{N}_eta{args.eta:g}{args.tag}.npz'
out_spec = HERE / 'exp_data' / f'phasefield_spectra_N{N}_eta{args.eta:g}{args.tag}.npz'
e_check = rel_error(np.zeros_like(b)) * E0
print(f'Parseval check: sum E(k) / N^2 = {spectra["initial"].sum() / N ** 2 / 2:.6e}, ||e0||_K^2 = {e_check:.6e}')


def record(key, x):
    spectra[key] = spectrum(x)
    spectra_geo[key] = spectrum_geometric(x)


# ---------------------------------------------------------------- one-level baselines and MG preconditioners
for method in ([] if args.spectra_only else args.methods):
    t0 = time.time()
    hier.setup(method)
    B = hier.preconditioner(method)
    mon = Monitor()
    pcg(fine, b, np.zeros_like(b), B, 100000, monitor=mon)
    results[method] = (np.array(mon.work), np.array(mon.error))
    print(f'{method:16s} it {mon.it:5d}  work {mon.work[-1]:8.1f}  err {mon.error[-1]:.1e}  {time.time() - t0:.0f} s',
          flush=True)

# spectra after n Green-CG iterations and after n Chebyshev-Jacobi smoothing steps (classical smoother)
for n in (2, 5, 20, 100):
    record(f'Green-CG {n}', pcg(fine, b, np.zeros_like(b), lambda r: fine.M('green', r), n))
for n in (2, 5, 20):
    record(f'GJ-CG {n}', pcg(fine, b, np.zeros_like(b), lambda r: fine.M('green_jacobi', r), n))
hier.setup('mg')
x_j = np.zeros_like(b)
for n in range(1, 6):
    x_j = hier.smooth(0, x_j, b, 'jacobi')
record('Chebyshev-Jacobi 5x2', x_j)
x_v = hier.vcycle(b, 0, 'jacobi')
record('MG V-cycle 1', x_v)
record('MG V-cycle 2', x_v + hier.vcycle(b - fine.K(x_v), 0, 'jacobi'))

# ---------------------------------------------------------------- the user's two-grid Green cycles
G_f = lambda r: fine.M('green', r)
G_c = lambda r: coarse.M('green', r)
for pair in (args.cycles if not args.spectra_only else ['2,2', '5,5', '5,20']):
    nu, nu_c = (int(v) for v in pair.split(','))
    t0 = time.time()
    mon = Monitor()
    x = np.zeros_like(b)
    cycle = 0
    while True:
        x = pcg(fine, b, x, G_f, nu, monitor=mon)                       # fine Green-CG
        if mon.error[-1] <= args.tol ** 2 or mon.work[-1] > args.max_work:
            break
        if cycle == 0:
            record(f'2-grid {nu},{nu_c}: after fine', x)
        r_H = hier.R(0, b - fine.K(x))
        e_H = pcg(coarse, r_H, np.zeros_like(r_H), G_c, nu_c)           # coarse Green-CG on 128
        x = x + hier.P(0, e_H)
        mon.work.append(work())
        mon.error.append(rel_error(x))
        if cycle == 0:
            record(f'2-grid {nu},{nu_c}: after coarse', x)
        cycle += 1
        if args.spectra_only:
            break
    results[f'green2grid {nu},{nu_c}'] = (np.array(mon.work), np.array(mon.error))
    print(f'green2grid {nu},{nu_c:<4d} cycles {cycle:4d} fine it {mon.it:5d}  work {mon.work[-1]:8.1f}  '
          f'err {mon.error[-1]:.1e}  {time.time() - t0:.0f} s', flush=True)

np.savez(out_spec, spectra=np.array(spectra, dtype=object), spectra_geometric=np.array(spectra_geo, dtype=object))
print(out_spec)
if not args.spectra_only:
    np.savez(out, eta=args.eta, N=N, w_G=args.w_G, w_T=args.w_T, tol=args.tol,
             results=np.array(results, dtype=object), s_min=s_q.min())
    print(out)
