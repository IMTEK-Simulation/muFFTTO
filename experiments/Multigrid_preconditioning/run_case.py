"""
One case of the multigrid study: problem P1/P2/P3, contrast kappa, grid N, a list of preconditioners.

For every method a fresh hierarchy is built (setup cost counted separately), then the library CG solves the
periodic cell problem with E = e_x (x) e_x, stopping on the Meurant-Papez-Tichy lower estimate of the energy
error (relative 1e-6, adaptive delay, minimum 10).  The true relative energy error at the stop is measured
against a reference solution (dual, (r, z) < 1e-26), so every method is checked to stop at the same accuracy.

Writes exp_data/<problem>_N<N>_kappa1e<k>_<method>.npz

    python run_case.py --problem P1 --kappa-exp 3 --grid 64 --methods green dual mg bnn-green_jacobi
"""
import sys
import os
import argparse
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from muFFTTO import solvers

import mg_lib
import problems

HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--problem', choices=['P1', 'P2', 'P3'], required=True)
parser.add_argument('--kappa-exp', type=int, required=True)
parser.add_argument('--grid', type=int, default=64)
parser.add_argument('--methods', nargs='+', required=True)
parser.add_argument('--n-coarsest', type=int, default=4)
parser.add_argument('--degree', type=int, default=2, help='Chebyshev smoothing steps (pre = post)')
parser.add_argument('--alpha', type=float, default=30.0, help='smoothing range [lmax / alpha, lmax]')
parser.add_argument('--tol', type=float, default=1e-6)
parser.add_argument('--d-min', type=int, default=10)
parser.add_argument('--maxiter', type=int, default=5000)
parser.add_argument('--tag', default='', help='suffix of the output file (parameter variants)')
args = parser.parse_args()

kappa = 10.0 ** args.kappa_exp
N = args.grid
C_q = problems.quad_material(args.problem, kappa, N)
out_dir = HERE / 'exp_data'
out_dir.mkdir(exist_ok=True)


def build():
    return mg_lib.Hierarchy(C_q, problems.C_0, n_coarsest=args.n_coarsest, degree=args.degree, alpha=args.alpha)


def system(hier):
    fine = hier.levels[0]
    disc = fine.disc
    macro = disc.get_gradient_size_field(name='macro_gradient_field')
    disc.get_macro_gradient_field_mugrid(macro_gradient_ij=np.array([[1.0, 0.0], [0.0, 0.0]]),
                                         macro_gradient_field_ijqxyz=macro)
    rhs = disc.get_unknown_size_field(name='rhs')
    disc.get_rhs_mugrid(material_data_field_ijklqxyz=fine.material, macro_gradient_field_ijqxyz=macro,
                        rhs_inxyz=rhs)

    def K_fun(x, Ax):
        fine._apply_K(fine.material, x, Ax)

    return disc, rhs, K_fun


def solve(hier, method, tol, stop, callback=None, **kw):
    disc, rhs, K_fun = system(hier)
    B = hier.preconditioner(method)

    def P_fun(r, z):
        z.s[...] = B(r.s.copy())

    x = disc.get_unknown_size_field(name='solution')
    x.s[...] = 0.0
    solvers.conjugate_gradients_mugrid_experimental(
        comm=disc.communicator, fc=disc.field_collection, hessp=K_fun, b=rhs, x=x, P=P_fun, tol=tol,
        maxiter=args.maxiter, callback=callback, stop_crit_norm=stop, **kw)
    return x.s.copy()


# ---------------------------------------------------------------- reference solution
ref_file = out_dir / f'reference_{args.problem}_N{N}_kappa1e{args.kappa_exp:+d}.npz'
if ref_file.exists():
    u_ref = np.load(ref_file)['u']
else:
    hier = build()
    hier.setup('dual')
    u_ref = solve(hier, 'dual', 1e-13, 'rz')
    np.savez(ref_file, u=u_ref)

for method in args.methods:
    t0 = time.time()
    hier = build()
    fine = hier.levels[0]
    t_build = time.time() - t0
    setup_cost = hier.setup(method)
    t_setup = time.time() - t0 - t_build
    energy_ref = np.sum(u_ref * fine.K(u_ref))                  # ||u_ref - 0||_K^2 = e_0
    hier.cost.update({'K': 0.0, 'G': 0.0, 'T': 0.0})
    hist = {'it': 0, 'error': [], 'cost': []}

    def count(it, x, r, p, z, stop_crit):
        hist['it'] = it
        hist['cost'].append(dict(hier.cost))
        e = u_ref - x
        hist['error'].append(np.sum(e * fine.K(e, count=False)) / energy_ref)

    t1 = time.time()
    u = solve(hier, method, args.tol, 'energy_lower_estim', callback=count, energy_rtol=True,
              d_min=args.d_min, tau=0.25)
    t_solve = time.time() - t1
    cost = dict(hier.cost)
    e = u_ref - u
    rel_err = np.sum(e * fine.K(e, count=False)) / energy_ref
    lev = [lv.N for lv in hier.levels]
    error = np.array(hist['error'])
    reached = np.nonzero(error <= args.tol ** 2)[0]
    k_star = int(reached[0]) if reached.size else -1
    cost_star = hist['cost'][k_star] if reached.size else {'K': np.nan, 'G': np.nan, 'T': np.nan}
    np.savez(out_dir / f'{args.problem}_N{N}_kappa1e{args.kappa_exp:+d}_{method}{args.tag}.npz',
             problem=args.problem, N=N, kappa=kappa, method=method, iterations=hist['it'],
             cost_K=cost['K'], cost_G=cost['G'], cost_T=cost['T'], error=error, k_star=k_star,
             star_K=cost_star['K'], star_G=cost_star['G'], star_T=cost_star['T'],
             setup_K=setup_cost['K'], setup_G=setup_cost['G'], rel_energy_error=rel_err,
             t_solve=t_solve, t_setup=t_setup, levels=lev, degree=args.degree, alpha=args.alpha,
             n_coarsest=args.n_coarsest, maxiter=args.maxiter)
    print(f'{args.problem} N{N} kappa 1e{args.kappa_exp:+d} {method:18s} it {hist["it"]:5d} k* {k_star:5d}  '
          f'K {cost["K"]:8.1f}  G {cost["G"]:8.1f}  T {cost["T"]:6.1f}  '
          f'|e|_K^2/|e0|^2 {rel_err:.1e}  {t_solve:6.1f} s', flush=True)
