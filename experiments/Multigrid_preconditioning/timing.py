"""
Relative cost of the building blocks on this machine (serial CPU): one operator application K, one Green
application G (FFT pair), one grid transfer (P or R, numpy).  Gives the weights w_G = t_G / t_K and
w_T = t_T / t_K of the work unit in summarize.py.

    python timing.py 64 128 256
"""
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import mg_lib
import problems

for N in [int(a) for a in sys.argv[1:]] or [64, 128, 256]:
    C_q = problems.quad_material('P1', 1e3, N)
    hier = mg_lib.Hierarchy(C_q, problems.C_0, n_coarsest=N // 2)
    lev = hier.levels[0]
    x = np.random.default_rng(0).standard_normal(lev.shape)
    xc = mg_lib.restrict(x)
    reps = max(20, int(2e6 / N ** 2))

    def best(f):
        f()
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                f()
            ts.append((time.perf_counter() - t0) / reps)
        return min(ts)

    t_K = best(lambda: lev.K(x))
    t_G = best(lambda: lev.M('green', x))
    t_T = 0.5 * (best(lambda: mg_lib.prolong(xc)) + best(lambda: mg_lib.restrict(x)))
    print(f'N = {N:4d}:  K {t_K * 1e3:7.3f} ms   G {t_G * 1e3:7.3f} ms   T {t_T * 1e3:7.3f} ms   '
          f'w_G = {t_G / t_K:.2f}   w_T = {t_T / t_K:.2f}', flush=True)
