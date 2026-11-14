"""
Table of the multigrid study:  iterations at the stop (energy estimate, adaptive delay >= 10),  k* (first
iteration with true ||e||_K^2 / ||e_0||_K^2 <= tol^2) and the solve work at the stop in fine-level operator
equivalents  W = K + w_G G + w_T T.

    python summarize.py [--grid 64] [--w-G 1.0] [--w-T 0.5] [--tag ""]
"""
import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--grid', type=int, default=64)
parser.add_argument('--w-G', type=float, default=1.0, help='cost of one Green application / one K application')
parser.add_argument('--w-T', type=float, default=0.5, help='cost of one grid transfer / one K application')
parser.add_argument('--tag', default='')
parser.add_argument('--metric', choices=['it', 'kstar', 'work'], default='it')
args = parser.parse_args()

METHODS = ['green', 'green_jacobi', 'dual', 'mg', 'mgS-green', 'mgS-green_jacobi', 'mgS-dual',
           'bnn-green_jacobi', 'bnn3-green', 'bnn3-green_jacobi', 'bnn3-dual', 'add-green_jacobi']
CASES = [('P1', k) for k in (-3, -2, -1, 1, 2, 3)] + [('P2', k) for k in (-3, -2, -1, 1, 2, 3)] + \
        [('P3', k) for k in (-1, -2, -3, -6)]


def load(problem, k, method):
    f = HERE / 'exp_data' / f'{problem}_N{args.grid}_kappa1e{k:+d}_{method}{args.tag}.npz'
    return np.load(f, allow_pickle=True) if f.exists() else None


print(f'N = {args.grid}, metric = {args.metric}  (work: w_G = {args.w_G}, w_T = {args.w_T})')
print(f'{"method":18s}' + ''.join(f'{p}/{k:+d}'.rjust(8) for p, k in CASES))
for method in METHODS:
    row = f'{method:18s}'
    for problem, k in CASES:
        d = load(problem, k, method)
        if d is None:
            row += '-'.rjust(8)
            continue
        if args.metric == 'it':
            v = int(d['iterations'])
            flag = '' if d['rel_energy_error'] <= 1e-12 else '!'
            row += f'{v}{flag}'.rjust(8)
        elif args.metric == 'kstar':
            row += f'{int(d["k_star"])}'.rjust(8)
        else:
            w = float(d['cost_K'] + args.w_G * d['cost_G'] + args.w_T * d['cost_T'])
            row += f'{w:.0f}'.rjust(8)
    print(row)
print('! = stopped with true relative energy error above tol^2')
