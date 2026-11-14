"""
Compare the objective history and the CG iterations of two TO logs (reference script or to3d_large.py):
    python fable_compare.py <reference log or fable_reference/*.txt> <log> [nb evaluations, default all common]
The reference script counts the CG callback calls (iterations + 1); to3d_large.py counts iterations.
"""
import re
import sys

import numpy as np


def read(path):
    f, mech, adj, wall = [], [], [], []
    for line in open(path):
        if line.startswith('EVAL'):
            parts = line.split()
            f.append(float(parts[3]))
            wall.append(float(parts[5]))
        elif 'nb_ steps CG mech' in line:
            mech.append(int(re.search(r'=\s*(\d+)', line).group(1)) - 1)
        elif 'nb_steps CG adjoint' in line:
            adj.append(int(re.search(r'=\s*(\d+)', line).group(1)) - 1)
        elif line.startswith('load case') and 'CG mechanical' in line:
            m = re.search(r'CG mechanical (\d+), adjoint (\d+)', line)
            mech.append(int(m.group(1)))
            adj.append(int(m.group(2)))
    return np.array(f), np.array(mech), np.array(adj), np.array(wall)


f_a, mech_a, adj_a, wall_a = read(sys.argv[1])
f_b, mech_b, adj_b, wall_b = read(sys.argv[2])
n = min(len(f_a), len(f_b)) if len(sys.argv) < 4 else int(sys.argv[3])
per_eval = len(mech_a) // len(f_a)
rel = np.abs(f_a[:n] - f_b[:n]) / np.abs(f_a[:n])
print(f'{n} evaluations compared; relative difference of f: max {rel.max():.1e}, per evaluation:')
print('  ' + ' '.join(f'{r:.0e}' for r in rel))
ca, cb = (mech_a + adj_a)[:n * per_eval], (mech_b + adj_b)[:n * (len(mech_b) // len(f_b))]
print(f'CG iterations in these evaluations: reference {ca.sum()} ({ca.sum() / n:.1f} per evaluation), '
      f'new {cb.sum()} ({cb.sum() / n:.1f} per evaluation)')
if len(wall_a) > 2 and len(wall_b) > 2:
    print(f's / evaluation (after the first): reference {(wall_a[n - 1] - wall_a[0]) / (n - 1):.2f}, '
          f'new {(wall_b[n - 1] - wall_b[0]) / (n - 1):.2f}')
