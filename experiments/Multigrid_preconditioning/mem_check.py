"""
Memory footprint per preconditioner (laptop CPU, one process per method): peak RSS increase over the baseline
(imports + geometry) for building what the method needs, its setup and 5 CG iterations.
    green, green_jacobi, dual   fine level only
    mg                          full hierarchy N, N/2, ..., 4
Material A = s C_0 stored as the full C_ijkl per quadrature point (muFFTTO material field).

    python mem_check.py <method> [--grid 256] [--eta 2]
"""
import sys
import os
import argparse
import resource

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mg_lib
import problems

parser = argparse.ArgumentParser()
parser.add_argument('method')
parser.add_argument('--grid', type=int, default=256)
args = parser.parse_args()
N = args.grid


def rss_MB():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


rng = np.random.default_rng(0)
s_q = 10.0 ** rng.uniform(-4, 0, size=(2, N, N))
_ = mg_lib.Hierarchy(np.ones((2, 8, 8)), problems.C_0, n_coarsest=8, scalar=True)   # load JAX / muGrid once
base = rss_MB()
hier = mg_lib.Hierarchy(s_q, problems.C_0, n_coarsest=4 if args.method == 'mg' else N, scalar=True)
built = rss_MB()
hier.setup(args.method)
B = hier.preconditioner(args.method)
fine = hier.levels[0]
b = rng.standard_normal(fine.shape)
x = np.zeros_like(b)
r = b.copy()
z = B(r)
p = z.copy()
rz = np.sum(r * z)
for _ in range(5):
    Ap = fine.K(p)
    alpha = rz / np.sum(p * Ap)
    x, r = x + alpha * p, r - alpha * Ap
    z = B(r)
    rz_new = np.sum(r * z)
    p, rz = z + rz_new / rz * p, rz_new
peak = rss_MB()
unknown_MB = 2 * N * N * 8 / 2 ** 20
print(f'N {N} {args.method:13s} levels {len(hier.levels)}  build +{built - base:7.1f} MB  peak +{peak - base:7.1f} MB'
      f'  = {(peak - base) / unknown_MB:5.1f} unknown-vectors  (1 vector = {unknown_MB:.2f} MB)', flush=True)
