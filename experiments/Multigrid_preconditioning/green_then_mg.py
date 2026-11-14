"""
User question 2026-11-13: "run a couple of CG iterations with Green and then a couple of multigrid?"
256^2 phase-field TO design (geometry as scaling_case.py), k = 1e-4, E_xx; work to the TRUE relative energy error
tol^2 (reference: dual-PCG to (r, z) < 1e-28), work = K + 0.2 G + 0.05 T (serial-CPU weights).

    a) mg            MG-PCG from zero
    b) green<k>+mg   k Green-PCG iterations, then MG-PCG started from that iterate
    c) alt 2+2       restarted: 2 Green-PCG, 2 MG-PCG, repeated
    d) VGV           PCG with one combined SPD preconditioner: V-cycle, Green correction, V-cycle
                     (symmetric multiplicative, I - BA = (I - VA)(I - GA)(I - VA))
    e) S-M-S         user idea 2: "multigrid on ONE grid, Green as the coarse correction": Chebyshev-Jacobi smoothing
                     (high frequencies) - global FFT correction M (low frequencies) - smoothing, M = Green, GJ
                     or dual scaled by 1 / lambda_max(M A) (GJ, dual; Green has lambda_max <= 1); SPD, PCG

    python green_then_mg.py [--grid 256] [--eta 2]
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

import mg_lib
import problems

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=256)
parser.add_argument('--eta', type=float, default=2.0)
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--variants', nargs='+', default=['mg', 'green_then', 'alt', 'VGV', 'SMS'])
args = parser.parse_args()
N, eta, k_soft = args.grid, args.eta, 1e-4

# ---------------------------------------------------------------- geometry (as scaling_case.py)
design = np.asarray(netCDF4.Dataset(HERE / 'data' / 'N256_soft0.0001_energy-warm_Green_eta_0.0078125_w_5.0_l_bfgs_final.nc')
                    ['phase_field_solution'][:]).squeeze()
solid = (design.reshape(N, 256 // N, N, 256 // N).mean(axis=(1, 3)) > 0.5) if N < 256 else \
    np.kron(design > 0.5, np.ones((N // 256, N // 256), dtype=bool))
big = np.tile(solid, (3, 3))
signed = np.where(solid, ndimage.distance_transform_edt(big)[N:2 * N, N:2 * N] - 0.5,
                  -(ndimage.distance_transform_edt(~big)[N:2 * N, N:2 * N] - 0.5))
phi = solid.astype(float) if eta == 0 else 0.5 * (1 + np.tanh(signed / (2 * eta)))
phi_x, phi_y = np.roll(phi, -1, axis=0), np.roll(phi, -1, axis=1)
phi_q = np.stack([(phi + phi_x + phi_y) / 3, (phi_x + phi_y + np.roll(phi_x, -1, axis=1)) / 3])
s_q = np.where(phi_q > 0.5, 1.0, k_soft) if eta == 0 else phi_q ** 2 * (1 - k_soft) + k_soft

hier = mg_lib.Hierarchy(s_q, problems.C_0, n_coarsest=4, scalar=True)
fine = hier.levels[0]
macro = fine.disc.get_gradient_size_field(name='macro_gradient_field')
fine.disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                          macro_gradient_field_ijqxyz=macro)
rhs_f = fine.disc.get_unknown_size_field(name='rhs')
fine.disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro,
                         rhs_inxyz=rhs_f)
b = rhs_f.s.copy()


def pcg(rhs, x, M, iters, monitor=None, rz_rtol=None):
    r = rhs.copy() if not np.any(x) else rhs - fine.K(x)
    z = M(r)
    p = z.copy()
    rz = rz0 = np.sum(r * z)
    for _ in range(iters):
        if rz_rtol is not None and rz < rz_rtol * rz0:
            break
        Ap = fine.K(p)
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


hier.setup('dual')
u_ref = pcg(b, np.zeros_like(b), lambda r: fine.M('dual', r), 20000, rz_rtol=1e-28)
E0 = np.sum(u_ref * fine.K(u_ref, count=False))
hier.setup('mg')


class Monitor:
    def __init__(self):
        hier.cost.update({'K': 0.0, 'G': 0.0, 'T': 0.0})
        self.it = 0
        self.err = 1.0

    def work(self):
        c = hier.cost
        return c['K'] + 0.2 * c['G'] + 0.05 * c['T']

    def __call__(self, x):
        self.it += 1
        e = u_ref - x
        self.err = np.sum(e * fine.K(e, count=False)) / E0
        return self.err <= args.tol ** 2 or self.work() > 3000


G = lambda r: fine.M('green', r)
V = lambda r: hier.vcycle(r, 0, 'jacobi')


def VGV(r):
    z = V(r)
    z = z + G(r - fine.K(z))
    return z + V(r - fine.K(z))


print(f'N {N}, eta/h {eta:g}: true relative energy error <= {args.tol ** 2:.0e}', flush=True)
print(f'{"variant":16s}{"iterations":>12s}{"work":>9s}{"error":>10s}')
zero = np.zeros_like(b)

mon = Monitor()
pcg(b, zero, V, 1000, mon)
print(f'{"a) mg":16s}{mon.it:12d}{mon.work():9.1f}{mon.err:10.1e}', flush=True)

for k in ((2, 5, 10, 20) if 'green_then' in args.variants else ()):
    mon = Monitor()
    x = pcg(b, zero, G, k, mon)
    k_green, w_green = mon.it, mon.work()
    pcg(b, x, V, 1000, mon)
    print(f'{f"b) green{k}+mg":16s}{f"{k_green} + {mon.it - k_green}":>12s}{mon.work():9.1f}{mon.err:10.1e}'
          f'   (Green part {w_green:.1f})', flush=True)

mon = Monitor()
x = zero
while 'alt' in args.variants and mon.err > args.tol ** 2 and mon.work() <= 3000:
    x = pcg(b, x, G, 2, mon)
    if mon.err <= args.tol ** 2:
        break
    x = pcg(b, x, V, 2, mon)
if 'alt' in args.variants:
    print(f'{"c) alt 2+2":16s}{mon.it:12d}{mon.work():9.1f}{mon.err:10.1e}', flush=True)

if 'VGV' in args.variants:
    mon = Monitor()
    pcg(b, zero, VGV, 1000, mon)
    print(f'{"d) VGV":16s}{mon.it:12d}{mon.work():9.1f}{mon.err:10.1e}', flush=True)

for kind in (('green', 'green_jacobi', 'dual') if 'SMS' in args.variants else ()):
    fine.setup(kind)
    omega = 1.0 if kind == 'green' else 1.0 / fine.lmax(kind)

    def SMS(r):
        x = hier.smooth(0, np.zeros_like(r), r, 'jacobi', x_is_zero=True)
        x = x + omega * fine.M(kind, r - fine.K(x))
        return hier.smooth(0, x, r, 'jacobi')

    for name, P in [(f'one-level {kind}', lambda r: fine.M(kind, r)), (f'e) S-{kind}-S', SMS)]:
        mon = Monitor()
        pcg(b, zero, P, 5000, mon)
        print(f'{name:22s}{mon.it:6d}{mon.work():9.1f}{mon.err:10.1e}', flush=True)
