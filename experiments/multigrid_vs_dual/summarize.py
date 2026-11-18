"""Table of the elastic-solve cost per run from the iterations CSVs (exp_data/*_iterations.csv).

usage: python summarize.py [pattern]      (default: every CSV in exp_data/)

Per run: elastic solves, mean / max CG iterations per solve, total elastic wall time (incl. the MG setup), its MG
setup share, K-equivalents per solve (stiffness applications in fine-grid equivalents, solve + setup) and Green (FFT
pair) applications per solve.
"""
import glob
import os
import sys

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
pattern = sys.argv[1] if len(sys.argv) > 1 else '*'
files = sorted(glob.glob(os.path.join(here, 'exp_data', f'{pattern}_iterations.csv')))
print(f'{"run":58s} {"solves":>6s} {"CG/solve":>8s} {"max":>5s} {"t_u [s]":>8s} {"setup":>6s} '
      f'{"K/solve":>8s} {"G/solve":>8s} {"t_d [s]":>8s}')
for f in files:
    data = np.genfromtxt(f, delimiter=',', names=True, dtype=None, encoding=None)
    if data.size == 0:
        continue
    data = np.atleast_1d(data)
    n = len(data)
    name = os.path.basename(f).replace('_iterations.csv', '')
    setup = data['time_setup_u'].sum() / max(data['time_u'].sum(), 1e-30)
    k = (data['kwork_u'] + data['kwork_setup_u']).sum() / n
    print(f'{name:58s} {n:6d} {data["cg_u"].mean():8.1f} {data["cg_u"].max():5d} {data["time_u"].sum():8.1f} '
          f'{setup:6.0%} {k:8.1f} {data["gwork_u"].sum() / n:8.1f} {data["time_d"].sum():8.1f}')
