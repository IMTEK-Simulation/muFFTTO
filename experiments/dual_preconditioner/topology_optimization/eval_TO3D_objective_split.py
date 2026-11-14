"""Split the logged objective of 3D TO histories into the stress mismatch and the interface term, without new solves:
    f = a * S + f_pf,    S = sum_G |sigma_G - sigma*_G|^2 / |sigma*_G|^2     (6 load cases)
f_pf (pf_scale x phase-field bracket) depends on the design only and is recomputed here for every stored iterate with
the library function the optimizer uses; S = (f - f_pf) / a. Check: for a finished run, S from its homogenized
stresses vs targets (attributes of _final.nc) against S of its last history frame.
    python eval_TO3D_objective_split.py <history.nc> [...]      (CPU)  -> CSV next to each history:  _objsplit.csv
"""
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import netCDF4
import numpy as np

from muFFTTO import domain
from muFFTTO import topology_optimization

disc = {}


def discretization(N):
    if N not in disc:
        cell = domain.PeriodicUnitCell(domain_size=[1, 1, 1], problem_type='elasticity')
        disc[N] = domain.Discretization(cell=cell, nb_of_pixels_global=(N, N, N), discretization_type='finite_element',
                                        element_type='trilinear_hexahedron')
    return disc[N]


def stress_mismatch_final(fn):
    """S from the stored homogenized and target stresses of a _final.nc (None if not there)."""
    if not os.path.exists(fn):
        return None
    with netCDF4.Dataset(fn) as d:
        if 'homogenized_stresses' not in d.ncattrs():
            return None
        s = np.array(d.getncattr('homogenized_stresses')).reshape(-1, 3, 3)
        t = np.array(d.getncattr('target_stresses')).reshape(-1, 3, 3)
    return float(sum(np.sum((a - b) ** 2) / np.sum(b ** 2) for a, b in zip(s, t)))


for fn in sys.argv[1:]:
    with netCDF4.Dataset(fn) as d:
        pf = d['phase_field_iterate']
        obj = np.array(d['objective'][:])
        eta, a, pf_scale = float(d.getncattr('eta')), float(d.getncattr('stress_weight')), float(d.getncattr('pf_scale'))
        dw = float(d.getncattr('double_well_depth'))
        N = pf.shape[-1]
        D = discretization(N)
        field = D.get_scalar_field(name='rho_eval')
        rows = []
        for k in range(pf.shape[0]):
            field.s[0, 0] = np.asarray(pf[k, 0], dtype=float)
            f_pf = pf_scale * float(topology_optimization.objective_function_phase_field(
                discretization=D, phase_field_1nxyz=field, eta=eta, double_well_depth=dw))
            rows.append((k, obj[k], f_pf, (obj[k] - f_pf) / a))
    out = fn.replace('_history.nc', '_objsplit.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'objective', 'f_interface', 'stress_mismatch_S'])
        w.writerows(rows)
    S_fin = stress_mismatch_final(fn.replace('_history.nc', '_final.nc'))
    print(f'{os.path.basename(fn)[:90]}: a = {a:g}, last frame f = {rows[-1][1]:.4f} = a S + f_pf with '
          f'f_pf = {rows[-1][2]:.4f}, S = {rows[-1][3]:.4f}'
          + (f'   [check: S from final homogenized stresses = {S_fin:.4f}]' if S_fin is not None else ''))
