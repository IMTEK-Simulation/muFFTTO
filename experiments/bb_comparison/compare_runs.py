"""
Table of a set of runs of the contact drivers (one problem, different solvers / preconditioners): totals over the load
path and the deviation of the result from a reference run.

    nit      outer iterations (Newton steps / BB iterations)
    hessp    Hessian-vector products (inner CG iterations; 0 for BB)
    fev      gradient (and energy) evaluations
    work     hessp + fev: operator applications, each one material evaluation + B / B^T (+ one preconditioner
             application for every CG iteration / BB iteration)
    time     wall time of the whole load path
    dP       max over common load levels |P - P_ref| / max |P_ref| (force on the loaded boundary)
    du       max |u - u_ref| / max |u_ref| at the last increment

usage:  python compare_runs.py <reference run> <run> ...      (prints a markdown table; --csv file appends rows)
"""
import os
import re
import shutil
import sys
import tempfile

import numpy as np
import netCDF4

args = [a for a in sys.argv[1:] if not a.startswith('--')]
csv = sys.argv[sys.argv.index('--csv') + 1] if '--csv' in sys.argv else None
if csv:
    args.remove(csv)


def load(folder):
    tmp = os.path.join(tempfile.mkdtemp(), 'c.nc')
    shutil.copy(os.path.join(folder, 'contact_fracture.nc'), tmp)
    ds = netCDF4.Dataset(tmp)
    r = {k: np.asarray(ds[k][:]) for k in ('nb_outer', 'nb_hessp', 'crown_u', 'crown_force', 'elapsed_time', 'lam')}
    r['nb_fev'] = np.asarray(ds['nb_fev'][:]) if 'nb_fev' in ds.variables else np.zeros_like(r['nb_hessp'])
    r['u'] = np.asarray(ds['u_fluc'][-1])
    r['precond'] = ds.getncattr('mechanics_preconditioner')
    r['n'] = ds['phase'].shape[-1]
    return r


def solver_of(folder):
    m = re.search(r'_(newton_cg|bb\d(?:energy|gradient)?)_', folder)
    g = re.search(r'_gtol([0-9.e+-]+)_', folder)
    return (m.group(1) if m else 'tr_newton') + (f' (gtol {g.group(1)})' if g else '')


ref = load(args[0])
rows = []
for folder in args:
    try:
        r = load(folder)
    except (OSError, KeyError, IndexError) as e:
        rows.append((solver_of(folder), '?', '?', 'no data', '', '', '', '', '', '', '', os.path.basename(folder.rstrip('/'))))
        continue
    done = abs(r['lam'][-1] - 1.0) < 1e-9
    # force compared only at load levels both runs visited (interpolating between different increment grids
    # of the nonlinear force curve alone gives ~1e-3)
    i, j = np.nonzero(np.abs(r['lam'][:, None] - ref['lam'][None, :]) < 1e-9)
    dP = np.max(np.abs(r['crown_force'][i] - ref['crown_force'][j])) / max(np.max(np.abs(ref['crown_force'])), 1e-30)
    du = np.max(np.abs(r['u'] - ref['u'])) / max(np.max(np.abs(ref['u'])), 1e-30) if done else np.nan
    rows.append((solver_of(folder), r['precond'], r['n'], f'{len(r["lam"])}' + ('' if done else ' (stopped)'),
                 int(r['nb_outer'].sum()), int(r['nb_hessp'].sum()), int(r['nb_fev'].sum()),
                 int(r['nb_hessp'].sum() + r['nb_fev'].sum()), f'{r["elapsed_time"][-1]:.1f}', f'{dP:.1e}',
                 f'{du:.1e}', os.path.basename(folder.rstrip('/'))))
header = ('solver', 'preconditioner', 'N', 'increments', 'nit', 'hessp', 'fev', 'work', 'time [s]', 'dP', 'du')
print('| ' + ' | '.join(header) + ' |')
print('|' + '---|' * len(header))
for row in rows:
    print('| ' + ' | '.join(str(v) for v in row[:len(header)]) + ' |')
if csv:
    new = not os.path.exists(csv)
    with open(csv, 'a') as out:
        if new:
            out.write(','.join(header + ('run',)) + '\n')
        for row in rows:
            out.write(','.join(str(v) for v in row) + '\n')
