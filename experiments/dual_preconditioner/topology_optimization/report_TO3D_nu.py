"""Poisson ratios / stiffness of 3D TO designs from their _final.nc (homogenized and target stresses):
    python report_TO3D_nu.py <final.nc> ...
"""
import sys

import netCDF4
import numpy as np

from elastic_properties import properties, voigt_from_stresses

for fn in sys.argv[1:]:
    with netCDF4.Dataset(fn) as d:
        C = voigt_from_stresses(np.array(d.getncattr('homogenized_stresses')).reshape(6, 3, 3))
        T = voigt_from_stresses(np.array(d.getncattr('target_stresses')).reshape(6, 3, 3))
        rho = np.asarray(d['phase_field_solution'][0, 0])
        nit = d.getncattr('nb_iterations') if 'nb_iterations' in d.ncattrs() else '?'
    p, pt = properties(C), properties(T)
    S = sum(np.sum((C[:, k] - T[:, k]) ** 2) / np.sum(T[:, k] ** 2) for k in range(6))
    print(f"{fn.split('/')[-1][:80]:80s} it {nit}  solid {rho.mean():.3f}  nu12 {p['nu12']:+.3f} nu21 {p['nu21']:+.3f} "
          f"nu13 {p['nu13']:+.3f} nu23 {p['nu23']:+.3f}  C11 {C[0, 0]:.4f} (target {T[0, 0]:.4f})  "
          f"mu {p['mu']:.4f} (target {pt['mu']:.4f})")
