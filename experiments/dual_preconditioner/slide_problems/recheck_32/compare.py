"""Rerun of 2026-10-16 (recheck_32/runs, the local copy of the script) against the 32^2 regular-grid slide runs of 2026-10-13/15
(slides/runs*): mechanics CG total, increments, final stress, wall time per (slide, preconditioner).

    python compare.py
"""
import glob
import os

import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
# the 2026-10-13/15 runs: Green / Green-Jacobi in the slides folder of the example, the alpha = 0 set in this experiment
SLIDES = os.path.join(HERE, '..', '..', '..', '..', 'examples', 'phase_field_fracture_adapted_grid', 'exp_data', 'slides')
NOREG = os.path.join(HERE, '..', 'runs_2026-10-15')
KIND = {'2_trust_region (alpha = 0)': '_elastic_lin-void_', '3_contact': '_elastic_2', '4_fracture': '_inner1e-06_2'}
TAG = {'Green': 'G', 'Green_Jacobi': 'GJ', 'dual': 'D'}


def summary(folder):
    f = glob.glob(os.path.join(folder, '*.nc'))
    d = netCDF4.Dataset([p for p in f if 'contact_fracture' in os.path.basename(p)][0])
    P = np.asarray(d['mean_P'][:])
    return (len(d['lam'][:]), float(d['lam'][-1]), int(np.sum(d['nb_hessp'][:])), float(d['elapsed_time'][-1]),
            P[-1, 1, 1], float(d.alpha) if 'alpha' in d.ncattrs() else np.nan)


print('| slide | preconditioner | run | increments | final lam | CG total | wall [s] | P_yy end | alpha |')
print('|---|---|---|---|---|---|---|---|---|')
for slide, key in KIND.items():
    for prec, tag in TAG.items():
        new = glob.glob(os.path.join(HERE, 'runs', f'*_N32_Gc0.1_{tag}_inner1e-06*'))
        new = [p for p in new if key in os.path.basename(p) + '2' and (key != '_elastic_2' or 'lin-void' not in p)]
        root = os.path.join(NOREG, 'runs_noreg') if 'alpha' in slide else os.path.join(SLIDES, 'runs')
        old = glob.glob(os.path.join(root, f'*regular_N32_Gc0.1_{tag}_inner1e-06*'))
        old = [p for p in old if key in os.path.basename(p) + '2' and (key != '_elastic_2' or 'lin-void' not in p)]
        for label, paths in (('yesterday', old), ('rerun', new)):
            for p in sorted(paths):
                try:
                    n, lam, cg, wall, P, alpha = summary(p)
                    print(f'| {slide} | {prec} | {label} | {n} | {lam:.3f} | {cg:,} | {wall:.0f} | {P:.6e} | {alpha:g} |')
                except (IndexError, OSError) as e:
                    print(f'| {slide} | {prec} | {label} | ({os.path.basename(p)}: {e}) |')
